#!/usr/bin/env python3
"""Score a frozen JSONL of instruction pairs across one or more Qwen servers."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_pairs(path: Path) -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            candidate = row.get("candidate")
            reference = row.get("reference")
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError(f"{path}:{line_number}: invalid candidate")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"{path}:{line_number}: invalid reference")
            candidate = candidate.strip()
            reference = reference.strip()
            pair_id = instruction_pair_id(candidate, reference)
            if row.get("pair_id") != pair_id or pair_id in pairs:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate pair_id")
            pairs[pair_id] = (candidate, reference)
    if not pairs:
        raise ValueError(f"{path}: no instruction pairs")
    return pairs


def _load_existing(
    path: Path,
    *,
    pairs: Mapping[str, tuple[str, str]],
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
                or pairs.get(pair_id) != (candidate.strip(), reference.strip())
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
                record.get("judgment"), candidate=candidate, reference=reference
            )
            records[pair_id] = record
    return records


async def _judge_all(
    pairs: Mapping[str, tuple[str, str]],
    *,
    endpoints: Sequence[str],
    model_name: str,
    model_revision: str,
    output: Path,
    concurrency_per_endpoint: int,
    max_tokens: int,
    retries: int,
) -> dict[str, dict[str, Any]]:
    from openai import AsyncOpenAI

    records = _load_existing(
        output,
        pairs=pairs,
        model_name=model_name,
        model_revision=model_revision,
    )
    pending = [pair_id for pair_id in sorted(pairs) if pair_id not in records]
    print(
        f"instruction pairs: total={len(pairs)} resumed={len(records)} "
        f"pending={len(pending)} endpoints={len(endpoints)}",
        file=sys.stderr,
        flush=True,
    )
    if not pending:
        _canonicalize_output(output, records)
        return records

    clients = [
        AsyncOpenAI(
            api_key="EMPTY",
            base_url=endpoint.rstrip("/") + "/v1",
            timeout=120.0,
            max_retries=0,
        )
        for endpoint in endpoints
    ]
    semaphores = [asyncio.Semaphore(concurrency_per_endpoint) for _ in endpoints]
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed = 0

    async def judge_one(index: int, pair_id: str) -> None:
        nonlocal completed
        candidate, reference = pairs[pair_id]
        endpoint_index = index % len(clients)
        client = clients[endpoint_index]
        semaphore = semaphores[endpoint_index]
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
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                content = response.choices[0].message.content
                judgment, _ = parse_judgment_text(
                    content or "", candidate=candidate, reference=reference
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
            except Exception as exc:  # retry API, JSON, and semantic failures
                last_error = exc
                validation_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise RuntimeError(f"failed pair {pair_id} after {retries} attempts") from last_error

    try:
        await asyncio.gather(
            *(judge_one(index, pair_id) for index, pair_id in enumerate(pending))
        )
    finally:
        await asyncio.gather(*(client.close() for client in clients))
    _canonicalize_output(output, records)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", default=QWEN_JUDGE_MODEL_NAME)
    parser.add_argument("--model-revision", default=QWEN_JUDGE_MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--concurrency-per-endpoint", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args(argv)
    if (
        args.concurrency_per_endpoint <= 0
        or args.max_tokens <= 0
        or args.retries <= 0
    ):
        parser.error("concurrency, max tokens, and retries must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pairs_path = args.pairs.resolve(strict=True)
    pairs = _load_pairs(pairs_path)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    records = asyncio.run(
        _judge_all(
            pairs,
            endpoints=args.endpoint,
            model_name=args.model,
            model_revision=args.model_revision,
            output=output,
            concurrency_per_endpoint=args.concurrency_per_endpoint,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
    )
    distribution = Counter(
        validate_judgment(record["judgment"]) for record in records.values()
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
        "input": {
            "path": str(pairs_path),
            "sha256": _file_sha256(pairs_path),
            "unique_instruction_pairs": len(pairs),
        },
        "endpoint_count": len(args.endpoint),
        "concurrency_per_endpoint": args.concurrency_per_endpoint,
        "pairs": len(records),
        "score_distribution": {
            str(score): distribution.get(score, 0) for score in range(1, 6)
        },
        "output": str(output),
        "output_sha256": _file_sha256(output),
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
