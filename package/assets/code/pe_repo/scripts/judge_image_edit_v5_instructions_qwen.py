#!/usr/bin/env python3
"""Score matched image-edit-v5 instructions with a Qwen3.5-9B server."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_image_edit_v5_region_splits import (  # noqa: E402
    GROUND_TRUTH_SCHEMAS,
    PREDICTION_SCHEMAS,
    PREDICTION_VIEWS,
    V5_SCHEMA,
    _instruction_pairs_for_qwen,
    _rows_by_id,
)
from scripts.image_edit_instruction_qwen_judge import (  # noqa: E402
    JUDGMENT_JSON_SCHEMA,
    QWEN_GENERATION_SEED,
    QWEN_JUDGE_MODEL_NAME,
    QWEN_JUDGE_MODEL_REVISION,
    QWEN_JUDGE_VERSION,
    QWEN_PROMPT_VERSION,
    build_messages,
    instruction_pair_id,
    parse_judgment_text,
    validate_judgment,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonicalize_output(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for pair_id in sorted(records):
            handle.write(
                json.dumps(
                    records[pair_id],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def _load_existing(
    path: Path,
    *,
    allowed_pair_ids: set[str],
    model_name: str,
    model_revision: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            candidate = record.get("candidate")
            reference = record.get("reference")
            if not isinstance(candidate, str) or not isinstance(reference, str):
                raise ValueError(f"{path}:{line_number}: invalid instruction pair")
            pair_id = instruction_pair_id(candidate, reference)
            if (
                record.get("pair_id") != pair_id
                or pair_id not in allowed_pair_ids
                or pair_id in records
            ):
                raise ValueError(f"{path}:{line_number}: invalid or duplicate pair_id")
            model = record.get("model")
            if (
                record.get("judge_version") != QWEN_JUDGE_VERSION
                or not isinstance(model, Mapping)
                or model.get("name") != model_name
                or model.get("revision") != model_revision
            ):
                raise ValueError(f"{path}:{line_number}: incompatible judge configuration")
            validate_judgment(
                record.get("judgment"),
                candidate=candidate,
                reference=reference,
            )
            records[pair_id] = record
    return records


async def _judge_all(
    pairs: Sequence[tuple[str, str]],
    *,
    endpoint: str,
    model_name: str,
    model_revision: str,
    output: Path,
    concurrency: int,
    max_tokens: int,
    retries: int,
) -> dict[str, dict[str, Any]]:
    from openai import AsyncOpenAI

    output.parent.mkdir(parents=True, exist_ok=True)
    pair_by_id = {
        instruction_pair_id(candidate, reference): (candidate.strip(), reference.strip())
        for candidate, reference in pairs
    }
    records = _load_existing(
        output,
        allowed_pair_ids=set(pair_by_id),
        model_name=model_name,
        model_revision=model_revision,
    )
    pending = [
        (pair_id, *pair_by_id[pair_id])
        for pair_id in sorted(pair_by_id)
        if pair_id not in records
    ]
    print(
        f"instruction pairs: total={len(pair_by_id)} resumed={len(records)} pending={len(pending)}",
        file=sys.stderr,
        flush=True,
    )
    if not pending:
        _canonicalize_output(output, records)
        return records

    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=endpoint.rstrip("/") + "/v1",
        timeout=120.0,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed = 0

    async def judge_one(pair_id: str, candidate: str, reference: str) -> None:
        nonlocal completed
        validation_error: str | None = None
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=build_messages(
                            candidate,
                            reference,
                            validation_error=validation_error,
                        ),
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=max_tokens,
                        seed=QWEN_GENERATION_SEED,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "instruction_score_1to5",
                                "strict": True,
                                "schema": JUDGMENT_JSON_SCHEMA,
                            },
                        },
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                content = response.choices[0].message.content
                judgment, _ = parse_judgment_text(
                    content or "",
                    candidate=candidate,
                    reference=reference,
                )
                record = {
                    "schema_version": 1,
                    "judge_version": QWEN_JUDGE_VERSION,
                    "prompt_version": QWEN_PROMPT_VERSION,
                    "pair_id": pair_id,
                    "candidate": candidate,
                    "reference": reference,
                    "model": {"name": model_name, "revision": model_revision},
                    "generation": {
                        "thinking": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": max_tokens,
                        "seed": QWEN_GENERATION_SEED,
                        "structured_json": True,
                    },
                    "judgment": judgment,
                    "completed_at": _utc_now(),
                }
                async with write_lock:
                    records[pair_id] = record
                    with output.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                async with progress_lock:
                    completed += 1
                    if completed % 100 == 0 or completed == len(pending):
                        print(
                            f"completed {completed}/{len(pending)} new judgments",
                            file=sys.stderr,
                            flush=True,
                        )
                return
            except Exception as exc:  # retry API, JSON, and semantic validation failures
                last_error = exc
                validation_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise RuntimeError(f"failed pair {pair_id} after {retries} attempts") from last_error

    try:
        await asyncio.gather(*(judge_one(*item) for item in pending))
    finally:
        await client.close()
    _canonicalize_output(output, records)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=QWEN_JUDGE_MODEL_NAME)
    parser.add_argument("--model-revision", default=QWEN_JUDGE_MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-schema", choices=PREDICTION_SCHEMAS, default=V5_SCHEMA)
    parser.add_argument(
        "--ground-truth-schema", choices=GROUND_TRUTH_SCHEMAS, default=V5_SCHEMA
    )
    parser.add_argument("--prediction-view", choices=PREDICTION_VIEWS, default="raw")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.concurrency <= 0 or args.max_tokens <= 0 or args.retries <= 0:
        parser.error("--concurrency, --max-tokens, and --retries must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _, predictions = _rows_by_id(args.predictions.resolve(strict=True))
    _, ground_truth = _rows_by_id(args.ground_truth.resolve(strict=True))
    all_pairs = _instruction_pairs_for_qwen(
        predictions,
        ground_truth,
        prediction_schema=args.prediction_schema,
        ground_truth_schema=args.ground_truth_schema,
        prediction_view=args.prediction_view,
    )
    pairs = [
        pair
        for index, pair in enumerate(all_pairs)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    output = args.output.resolve()
    started_at = _utc_now()
    records = asyncio.run(
        _judge_all(
            pairs,
            endpoint=args.endpoint,
            model_name=args.model,
            model_revision=args.model_revision,
            output=output,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
    )
    distribution = Counter(
        validate_judgment(record["judgment"])
        for record in records.values()
    )
    summary = {
        "status": "complete",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "judge_version": QWEN_JUDGE_VERSION,
        "prompt_version": QWEN_PROMPT_VERSION,
        "model": {"name": args.model, "revision": args.model_revision},
        "generation": {
            "thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": QWEN_GENERATION_SEED,
            "structured_json": True,
        },
        "sharding": {
            "all_pairs": len(all_pairs),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "selection": "sorted-pair-index-modulo",
        },
        "pairs": len(records),
        "score_distribution": {str(score): distribution.get(score, 0) for score in range(1, 6)},
        "output": str(output),
    }
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output
        else output.with_suffix(".summary.json")
    )
    _write_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
