#!/usr/bin/env python3
"""Evaluate end-to-end F1 format compliance with real GPT web searches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from agent_sft.agent_models import get_agent_model_adapter
from agent_sft.evaluation.format_compliance import (
    aggregate_trajectory_results,
    check_final_turn,
    check_think_format,
    check_tool_result,
    check_tool_turn,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_rank(seed: int, category: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{sample_id}".encode()).hexdigest()


def select_samples(
    rows: Iterable[dict[str, Any]],
    *,
    samples_per_category: int,
    seed: int,
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = str(row["metadata"]["subcategory"])
        if categories is not None and category not in categories:
            continue
        grouped.setdefault(category, []).append(row)
    if categories is not None:
        missing = categories - set(grouped)
        if missing:
            raise ValueError(f"unknown or empty categories: {sorted(missing)}")
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        ranked = sorted(
            grouped[category],
            key=lambda row: _stable_rank(seed, category, str(row["sample_id"])),
        )
        if len(ranked) < samples_per_category:
            raise ValueError(
                f"{category} has {len(ranked)} rows, fewer than {samples_per_category}"
            )
        selected.extend(ranked[:samples_per_category])
    return selected


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _extract_sources(raw_response: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for node in _walk(raw_response):
        if not isinstance(node, dict) or node.get("type") != "url_citation":
            continue
        url = node.get("url")
        if not isinstance(url, str) or not url.strip() or url in seen:
            continue
        title = node.get("title")
        sources.append(
            {
                "title": title.strip()
                if isinstance(title, str) and title.strip()
                else "Untitled source",
                "url": url,
            }
        )
        seen.add(url)
    return sources


def _extract_search_actions(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": item.get("id"), "status": item.get("status"), "action": item.get("action")}
        for item in raw_response.get("output", [])
        if isinstance(item, dict) and item.get("type") == "web_search_call"
    ]


def execute_gpt_web_search(
    client: Any,
    *,
    model: str,
    round_index: int,
    query: str,
    user_request: str,
    prior_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prior_context = [
        {
            "round": item["round"],
            "query": item["query"],
            "search_text": item["search_text"],
        }
        for item in prior_results
    ]
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[{"type": "web_search", "search_context_size": "medium"}],
        input=(
            "Execute one live web-search round for a text-to-image prompt-enhancement "
            "agent. Search rather than answering from memory. Prefer primary or "
            "authoritative sources and return a concise evidence report with citations.\n\n"
            f"USER REQUEST:\n{user_request}\n\n"
            f"CURRENT QUERY (round {round_index}):\n{query}\n\n"
            "PRIOR SEARCH RESULTS:\n"
            f"{json.dumps(prior_context, ensure_ascii=False, indent=2)}"
        ),
        max_output_tokens=8_000,
        store=False,
    )
    raw = response.model_dump(mode="json")
    actions = _extract_search_actions(raw)
    if not actions:
        raise RuntimeError("GPT response did not execute a web_search action")
    search_text = response.output_text
    if not isinstance(search_text, str) or not search_text.strip():
        raise RuntimeError("GPT web_search returned empty search_text")
    result = {
        "round": round_index,
        "query": query,
        "search_text": search_text,
        "sources": _extract_sources(raw),
    }
    audit = {
        "model": model,
        "actions": actions,
        "usage": raw.get("usage"),
        "response_id": raw.get("id"),
    }
    return result, audit


def _user_request(row: dict[str, Any]) -> str:
    return next(
        str(message["content"]) for message in row["messages"] if message["role"] == "user"
    )


def _initial_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for message in row["messages"]:
        if message["role"] == "assistant":
            break
        if message["role"] in {"system", "user"}:
            messages.append({"role": message["role"], "content": message["content"]})
    if [message["role"] for message in messages] != ["system", "user"]:
        raise ValueError(f"{row['sample_id']}: expected one initial system/user pair")
    return messages


def _failure_reasons(prefix: str, report: dict[str, Any]) -> list[str]:
    return [f"{prefix}:{failure}" for failure in report["failures"]]


def run_trajectory(
    engine: Any,
    client: Any,
    row: dict[str, Any],
    *,
    search_model: str,
    max_search_rounds: int,
    request_config: Any,
) -> dict[str, Any]:
    from swift.infer_engine import InferRequest

    adapter = get_agent_model_adapter("qwen3_5")
    messages = _initial_messages(row)
    tools = row["tools"]
    queries: list[str] = []
    search_results: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    failures: list[str] = []
    think_reports: list[dict[str, Any]] = []
    tool_reports: list[dict[str, Any]] = []
    execution_reports: list[dict[str, Any]] = []
    final_report: dict[str, Any] | None = None
    terminated = False
    transition_ok = True

    for turn_index in range(max_search_rounds + 1):
        response = engine.infer(
            [
                InferRequest(
                    messages=messages,
                    tools=tools,
                    chat_template_kwargs={"enable_thinking": True},
                )
            ],
            request_config,
            use_tqdm=False,
        )[0]
        choice = response.choices[0]
        text = str(choice.message.content)
        parser_valid, calls = adapter.parse_tool_calls(text)
        think_report = check_think_format(text)
        think_reports.append(think_report)
        failures.extend(_failure_reasons(f"turn_{turn_index + 1}.think", think_report))
        turn: dict[str, Any] = {
            "turn": turn_index + 1,
            "response": text,
            "finish_reason": choice.finish_reason,
            "usage": asdict(response.usage),
            "think": think_report,
            "parsed_tool_calls": [asdict(call) for call in calls],
        }
        turns.append(turn)
        tool_syntax_present = "<tool_call>" in text or "</tool_call>" in text

        if calls or tool_syntax_present:
            tool_report = check_tool_turn(
                text,
                parser_valid=parser_valid,
                calls=calls,
                prior_queries=queries,
            )
            tool_reports.append(tool_report)
            turn["tool_call"] = tool_report
            failures.extend(_failure_reasons(f"turn_{turn_index + 1}.tool", tool_report))
            if not tool_report["compliant"]:
                transition_ok = False
                break
            if len(queries) >= max_search_rounds:
                failures.append("trajectory:max_search_rounds_exceeded")
                transition_ok = False
                break

            query = str(tool_report["query"])
            call_id = f"web_{row['sample_id'].split(':')[-1]}_{len(queries) + 1}"
            assistant_content = text.split("<tool_call>", 1)[0].rstrip()
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": query}, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            try:
                tool_result, search_audit = execute_gpt_web_search(
                    client,
                    model=search_model,
                    round_index=len(queries) + 1,
                    query=query,
                    user_request=_user_request(row),
                    prior_results=search_results,
                )
            except Exception as exc:
                failures.append(f"turn_{turn_index + 1}.execution:{type(exc).__name__}")
                turn["search_error"] = {"type": type(exc).__name__, "message": str(exc)}
                transition_ok = False
                break

            result_report = check_tool_result(
                tool_result,
                expected_round=len(queries) + 1,
                expected_query=query,
            )
            execution_reports.append(result_report)
            turn["tool_result"] = tool_result
            turn["search_audit"] = search_audit
            turn["tool_result_format"] = result_report
            failures.extend(
                _failure_reasons(f"turn_{turn_index + 1}.execution", result_report)
            )
            if not result_report["compliant"]:
                transition_ok = False
                break
            queries.append(query)
            search_results.append(tool_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
            continue

        final_report = check_final_turn(text, parser_valid=parser_valid, calls=calls)
        turn["final"] = final_report
        failures.extend(_failure_reasons(f"turn_{turn_index + 1}.final", final_report))
        terminated = True
        break

    if not queries:
        failures.append("trajectory:no_real_tool_execution")
    if not terminated:
        failures.append("trajectory:no_final_response")
    if len(queries) > max_search_rounds:
        failures.append("trajectory:too_many_searches")
    metrics = {
        "think_compliant": bool(think_reports)
        and all(report["compliant"] for report in think_reports),
        "tool_call_compliant": bool(tool_reports)
        and all(report["compliant"] for report in tool_reports),
        "tool_execution_compliant": bool(execution_reports)
        and len(execution_reports) == len(tool_reports)
        and all(report["compliant"] for report in execution_reports),
        "transition_compliant": transition_ok
        and terminated
        and 1 <= len(queries) <= max_search_rounds,
        "final_schema_compliant": final_report is not None and final_report["compliant"],
    }
    metrics["overall_compliant"] = all(metrics.values())
    return {
        "sample_id": row["sample_id"],
        "category": row["metadata"]["subcategory"],
        "user_request": _user_request(row),
        "metrics": metrics,
        "failure_reasons": list(dict.fromkeys(failures)),
        "search_count": len(queries),
        "queries": queries,
        "turns": turns,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=1)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument(
        "--sample-id",
        help="Evaluate exactly one validation row by sample_id, bypassing category sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-search-rounds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--attn-impl",
        default="eager",
        help="Transformers attention implementation; eager avoids cuDNN SDPA ABI issues.",
    )
    parser.add_argument("--search-model", default=os.getenv("AGENT_SFT_SEARCH_MODEL", "gpt-5.6"))
    parser.add_argument("--revision")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.samples_per_category <= 0:
        parser.error("--samples-per-category must be positive")
    if args.max_search_rounds <= 0:
        parser.error("--max-search-rounds must be positive")
    if args.artifact.exists() and not args.overwrite:
        parser.error(f"artifact already exists: {args.artifact}")
    if not args.validation_file.is_file():
        parser.error(f"missing validation file: {args.validation_file}")
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required for real GPT web search")

    from openai import OpenAI
    from swift.infer_engine import RequestConfig, TransformersEngine

    rows = _read_jsonl(args.validation_file)
    if args.sample_id:
        selected = [row for row in rows if str(row.get("sample_id")) == args.sample_id]
        if len(selected) != 1:
            parser.error(
                f"--sample-id must match exactly one validation row; "
                f"found {len(selected)}: {args.sample_id}"
            )
    else:
        selected = select_samples(
            rows,
            samples_per_category=args.samples_per_category,
            seed=args.seed,
            categories=set(args.categories) if args.categories else None,
        )
    engine = TransformersEngine(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
        template_type="qwen3_5",
        use_hf=True,
        revision=args.revision,
        max_batch_size=1,
    )
    client = OpenAI(timeout=180.0)
    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        seed=args.seed,
        return_details=False,
    )
    results = []
    for index, row in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {row['metadata']['subcategory']} "
            f"{row['sample_id']}",
            flush=True,
        )
        result = run_trajectory(
            engine,
            client,
            row,
            search_model=args.search_model,
            max_search_rounds=args.max_search_rounds,
            request_config=request_config,
        )
        results.append(result)
        partial = {
            "schema_version": 1,
            "status": "running",
            "model": args.model,
            "search_model": args.search_model,
            "validation_file": str(args.validation_file.resolve()),
            "seed": args.seed,
            "samples_per_category": args.samples_per_category,
            "selected_sample_ids": [item["sample_id"] for item in selected],
            "summary": aggregate_trajectory_results(results),
            "results": results,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(args.artifact, partial)

    artifact = {
        "schema_version": 1,
        "status": "completed",
        "model": args.model,
        "search_model": args.search_model,
        "validation_file": str(args.validation_file.resolve()),
        "seed": args.seed,
        "samples_per_category": args.samples_per_category,
        "selected_sample_ids": [item["sample_id"] for item in selected],
        "summary": aggregate_trajectory_results(results),
        "results": results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.artifact, artifact)
    print(json.dumps(artifact["summary"]["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
