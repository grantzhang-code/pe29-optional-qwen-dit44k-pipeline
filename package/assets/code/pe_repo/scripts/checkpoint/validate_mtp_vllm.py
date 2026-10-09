#!/usr/bin/env python3
"""Run a bounded vLLM MTP generation and report speculative-decoding metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _metric_records(metrics: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in metrics:
        family_name = str(getattr(family, "name", ""))
        samples = getattr(family, "samples", None)
        if samples is None:
            if "spec_decode" in family_name or "speculative" in family_name:
                value = getattr(family, "value", None)
                if value is None and hasattr(family, "values"):
                    value = getattr(family, "values")
                records.append(
                    {
                        "name": family_name,
                        "value": _json_safe(value),
                        "labels": _json_safe(getattr(family, "labels", None)),
                    }
                )
            continue
        for sample in samples:
            name = str(getattr(sample, "name", ""))
            if "spec_decode" not in name and "speculative" not in name:
                continue
            records.append(
                {
                    "name": name,
                    "value": _json_safe(getattr(sample, "value", None)),
                    "labels": _json_safe(getattr(sample, "labels", None)),
                }
            )
    return records


def _has_positive_drafts(records: list[dict[str, Any]]) -> bool:
    return any(
        "num_drafts" in record["name"] and isinstance(record["value"], (int, float)) and record["value"] > 0
        for record in records
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Explain in one short sentence why layout hierarchy matters.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model = args.model.resolve()
    print(json.dumps({"model": str(model), "phase": "start"}, sort_keys=True), flush=True)

    # Import inside the real file entrypoint: vLLM uses multiprocessing "spawn"
    # after CUDA initialization and cannot safely relaunch an stdin/<string> main.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model),
        tokenizer=str(model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        attention_backend=args.attention_backend,
        speculative_config={
            "attention_backend": args.attention_backend,
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        disable_log_stats=False,
    )
    engine = llm.llm_engine
    speculative = engine.vllm_config.speculative_config
    if speculative is None:
        raise RuntimeError("vLLM did not construct a speculative-decoding config")
    draft_architectures = speculative.draft_model_config.hf_config.architectures or []
    if speculative.method != "mtp" or "Qwen3_5MTP" not in draft_architectures:
        raise RuntimeError(
            "vLLM did not resolve the Qwen3.5 MTP drafter: "
            f"method={speculative.method!r} architectures={draft_architectures!r}"
        )
    print(
        json.dumps(
            {
                "draft_architectures": draft_architectures,
                "num_speculative_tokens": speculative.num_speculative_tokens,
                "phase": "loaded",
                "spec_method": speculative.method,
                "target_architectures": engine.model_config.hf_config.architectures,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    print(
        json.dumps(
            {"phase": "generated", "text": outputs[0].outputs[0].text},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    records = _metric_records(llm.get_metrics())
    print(json.dumps({"metrics": records, "phase": "metrics"}, sort_keys=True), flush=True)
    if not _has_positive_drafts(records):
        raise RuntimeError("vLLM reported no positive speculative draft-token metric")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
