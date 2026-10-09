#!/usr/bin/env python3
"""Verify F1 supervision masks with the installed Qwen3.5 SWIFT template."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from agent_sft.config import dataset_fingerprint
from swift.model import get_model_processor
from swift.template import get_template


def _load_manifest(dataset_config_path: Path) -> tuple[dict[str, Any], Path]:
    config = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8"))
    fingerprint = dataset_fingerprint(config)
    root = Path(config["prepared_root"]).expanduser().resolve()
    prepared = root / f"{config['id']}-{fingerprint}"
    return json.loads((prepared / "manifest.json").read_text(encoding="utf-8")), prepared


def _label_spans(
    processor: Any,
    input_ids: list[int],
    labels: list[int],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    start = 0
    trained = labels[0] != -100
    for index, label in enumerate(labels[1:], 1):
        next_trained = label != -100
        if next_trained == trained:
            continue
        spans.append(
            {
                "start": start,
                "end": index,
                "trained": trained,
                "text": processor.decode(input_ids[start:index], skip_special_tokens=False),
            }
        )
        start = index
        trained = next_trained
    spans.append(
        {
            "start": start,
            "end": len(input_ids),
            "trained": trained,
            "text": processor.decode(input_ids[start:], skip_special_tokens=False),
        }
    )
    return spans


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest, prepared = _load_manifest(args.dataset_config.resolve())
    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    _, processor = get_model_processor(
        model_config["model_id"],
        load_model=False,
        use_hf=bool(model_config.get("megatron_args", {}).get("use_hf", True)),
        revision=model_config["revision"],
    )
    template = get_template(
        processor,
        template_type=model_config["template"],
        agent_template=model_config["agent_template"],
        loss_scale=model_config["megatron_args"]["loss_scale"],
        max_length=args.max_length,
        truncation_strategy="raise",
        add_non_thinking_prefix=model_config["megatron_args"].get("add_non_thinking_prefix", True),
    )
    template.set_mode("train")

    by_category: dict[str, dict[str, Any]] = {}
    test_path = prepared / manifest["files"]["evaluation"]
    for line in test_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        category = record["metadata"]["subcategory"]
        by_category.setdefault(category, record)

    totals: defaultdict[str, int] = defaultdict(int)
    samples: list[dict[str, Any]] = []
    for category in sorted(by_category):
        canonical = by_category[category]
        swift_record = {
            "messages": [],
            "tools": canonical["tools"],
        }
        for message in canonical["messages"]:
            calls = message.get("tool_calls") or []
            if message["role"] == "assistant" and calls:
                if message.get("content"):
                    swift_record["messages"].append(
                        {"role": "assistant", "content": message["content"], "loss": True}
                    )
                for call in calls:
                    function = call["function"]
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    swift_record["messages"].append(
                        {
                            "role": "tool_call",
                            "content": json.dumps(
                                {"name": function["name"], "arguments": arguments},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            "loss": True,
                        }
                    )
            elif message["role"] == "assistant":
                swift_record["messages"].append(
                    {"role": "assistant", "content": message["content"], "loss": True}
                )
            elif message["role"] == "tool":
                swift_record["messages"].append(
                    {"role": "tool", "content": message["content"], "loss": False}
                )
            else:
                swift_record["messages"].append(
                    {"role": message["role"], "content": message["content"], "loss": False}
                )

        encoded = template.encode(swift_record)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        spans = _label_spans(processor, input_ids, labels)
        trained_text = "".join(span["text"] for span in spans if span["trained"])
        masked_text = "".join(span["text"] for span in spans if not span["trained"])
        rendered_text = trained_text + masked_text
        user_message = next(
            message for message in canonical["messages"] if message["role"] == "user"
        )
        assertions = {
            "system_tools_masked": "# Tools" in masked_text,
            "system_prompt_masked": (
                "advanced text-to-image prompt enhancement agent" in masked_text
                and "Current local context:" in masked_text
            ),
            "user_masked": user_message["content"] in masked_text,
            "tool_response_masked": "<tool_response>" in masked_text,
            "assistant_reasoning_trained": "<think>" in trained_text,
            "stage_tags_removed": not any(
                tag in rendered_text
                for tag in (
                    "post_user_cot",
                    "pre_tool_cot",
                    "search_refinement_cot",
                    "post_tool_cot",
                    "pre_enhance_cot",
                )
            ),
            "tool_call_trained": "<function=web_search>" in trained_text,
            "internal_tool_arguments_removed": "target_constraint_ids" not in rendered_text,
            "final_answer_trained": '"enhanced_prompt"' in trained_text,
        }
        if not all(assertions.values()):
            raise RuntimeError(f"{category} mask verification failed: {assertions}")
        trained_tokens = sum(label != -100 for label in labels)
        masked_tokens = len(labels) - trained_tokens
        totals["samples"] += 1
        totals["tokens"] += len(labels)
        totals["trained_tokens"] += trained_tokens
        totals["masked_tokens"] += masked_tokens
        samples.append(
            {
                "sample_id": canonical["sample_id"],
                "category": category,
                "tokens": len(labels),
                "trained_tokens": trained_tokens,
                "masked_tokens": masked_tokens,
                "assertions": assertions,
                "spans": [
                    {
                        "start": span["start"],
                        "end": span["end"],
                        "trained": span["trained"],
                        "preview": span["text"][:240],
                    }
                    for span in spans
                ],
            }
        )

    report = {
        "passed": True,
        "model_id": model_config["model_id"],
        "model_revision": model_config["revision"],
        "template": model_config["template"],
        "agent_template": model_config["agent_template"],
        "loss_scale": model_config["megatron_args"]["loss_scale"],
        "prepared_dataset": str(prepared),
        "totals": dict(totals),
        "samples": samples,
    }
    output = prepared / "token_mask_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output"] = str(output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/datasets/f1_websearch_gemini_reverse.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/qwen3_5_9b.yaml"),
    )
    parser.add_argument("--max-length", type=int, default=32768)
    return parser


def main() -> None:
    print(json.dumps(verify(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
