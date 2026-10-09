#!/usr/bin/env python3
import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import common as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    rows = C.rows(args.rank)
    if args.limit:
        rows = rows[:args.limit]
    pending = []
    for row in rows:
        path = C.case_path('pe', row)
        if path.exists():
            C.validate_binding(json.loads(path.read_bytes()), row)
        else:
            pending.append(row)
    if not pending:
        return
    sys.path.insert(0, str(Path(__file__).parent / 'official'))
    import pe_core as core
    # Preserve strict model JSON and retry malformed answers explicitly.
    core.json_repair = None
    from vllm import LLM, SamplingParams
    model = C.MODEL_CACHE
    manifest = json.loads((model / 'verified-manifest.json').read_bytes())
    if manifest['revision'] != C.load_lock()['pe_model']['revision']:
        raise ValueError('PE model revision changed')
    C.progress(args.rank, 'pe_loading', total=len(rows))
    llm = LLM(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=32768, max_num_seqs=8, max_num_batched_tokens=32768,
              gpu_memory_utilization=0.80, enable_prefix_caching=False,
              limit_mm_per_prompt={'image': 1}, enforce_eager=True,
              trust_remote_code=False, seed=42)
    system = (model / 'system_prompt.txt').read_text().strip()
    completed = len(rows) - len(pending)
    for start in range(0, len(pending), 8):
        batch = pending[start:start + 8]
        messages, prompts = [], []
        for row in batch:
            image = C.source(row)
            w, h = image.size
            if w * h > 1048576:
                scale = (1048576 / (w * h)) ** 0.5
                from PIL import Image
                image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
            buf = BytesIO()
            image.save(buf, format='PNG')
            uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
            size = row['target_canvas_size']
            prompt = row['instruction'] + f"\n\nOutput canvas requirement: width={size['width']} px, height={size['height']} px (width × height). Use this output canvas when rewriting the edit instruction."
            prompts.append(prompt)
            messages.append(core.build_messages(system, prompt, [uri]))
        options = [SamplingParams(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                                  presence_penalty=0.0, max_tokens=24000, seed=42 + r['input_index']) for r in batch]
        outputs = llm.chat(messages, sampling_params=options, chat_template_kwargs={'enable_thinking': True})
        for row, prompt, conversation, out in zip(batch, prompts, messages, outputs):
            attempts = []
            for attempt in range(3):
                text = out.outputs[0].text
                thinking, answer = core.split_thinking(text)
                parsed = core.parse_answer(answer, core.PROFILES['edit'])
                attempts.append({'attempt': attempt, 'raw_text': text, 'parse_ok': parsed['parse_ok'],
                                 'finish_reason': out.outputs[0].finish_reason,
                                 'output_tokens': len(out.outputs[0].token_ids)})
                if parsed['parse_ok'] and out.outputs[0].finish_reason != 'length':
                    break
                if attempt == 2:
                    raise ValueError(f"PE answer invalid after three attempts: case {row['input_index']}")
                retry = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                                       presence_penalty=0.0, max_tokens=24000,
                                       seed=42 + row['input_index'] + (attempt + 1) * 1000000)
                out = llm.chat([conversation], sampling_params=retry, chat_template_kwargs={'enable_thinking': True})[0]
            record = C.binding(row) | {'status': 'complete', 'model': manifest['model'],
                'model_revision': manifest['revision'], 'model_manifest_sha256': C.file_sha(model / 'verified-manifest.json'),
                'system_prompt_sha256': C.file_sha(model / 'system_prompt.txt'), 'pe_user_prompt': prompt,
                'rewritten_prompt': parsed['positive_prompt'], 'wh_ratio': parsed['wh_ratio'],
                'ratio_follow': parsed['ratio_follow'], 'answer': answer,
                'attempts': attempts, 'worker_sha256': C.file_sha(__file__)}
            C.save_record('pe', row, record)
            completed += 1
            C.progress(args.rank, 'pe', completed, total=len(rows), last_index=row['input_index'])
    C.progress(args.rank, 'pe_complete', completed, total=len(rows))


if __name__ == '__main__':
    main()
