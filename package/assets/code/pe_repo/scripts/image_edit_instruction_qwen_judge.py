"""Pinned Qwen3.5-9B 1--5 semantic scoring for edit instructions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


QWEN_JUDGE_MODEL_NAME = "Qwen/Qwen3.5-9B"
QWEN_JUDGE_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN_JUDGE_VERSION = "qwen3.5-9b-instruction-score-1to5-v3"
QWEN_PROMPT_VERSION = "instruction-semantic-score-1to5-v3"
QWEN_GENERATION_SEED = 20260811

JUDGMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    },
    "required": ["score", "reason"],
}

SYSTEM_PROMPT = """You are a strict evaluator of image-edit instructions.

Compare the model-predicted CANDIDATE instruction with the human REFERENCE instruction. Judge semantic correctness and completeness, not lexical overlap. Evaluate only the text: do not infer image content, bounding boxes, or visual quality. Ignore generic dataset wording such as "referenced Source" and "in the Target" when it does not change the requested edit.

Check every explicit part of the edit plan: operation, target object, source/current state, desired attributes or state, count, literal text, size, style, material, orientation, spatial/geometry constraints, and preservation requirements. Penalize omissions, unsupported additions, and contradictions. Never claim that an omitted detail is implied by an unseen image or context.

Use exactly one integer score:
5 = Fully correct and semantically equivalent; every explicit reference detail is preserved and there is no unsupported detail. Faithful paraphrases and synonyms qualify.
4 = Mostly correct; the central edit is right, but there is one minor explicit omission, addition, or imprecision.
3 = Partially correct; the general edit is right, but a detail that changes the intended result is missing/wrong, or there are multiple minor issues.
2 = Weakly related; the operation, target, or several essential details differ, so the instruction would mostly fail the reference request.
1 = Unrelated, opposite, or fundamentally incompatible with the reference instruction.

Calibration: omitting "digital" from "digital wristwatch" is 4, changing "faint" to "bright" is at most 3, and asking to add a new object instead of expanding an existing one is at most 2.

Apply these mandatory upper bounds before returning the score:
- A different operation category (add/remove/replace/modify/preserve/move), or a different target object: score at most 2.
- A contradiction in an explicit desired value, count, literal text, or spatial relation: score at most 3.
- Any explicit detail omitted or unsupported detail added: score at most 4.
The score and reason must be logically consistent.

Return only the JSON object required by the response schema. The reason must be one sentence of at most 35 words."""


def instruction_pair_id(candidate: str, reference: str) -> str:
    payload = json.dumps(
        [candidate.strip(), reference.strip()],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_messages(
    candidate: str,
    reference: str,
    *,
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    candidate = candidate.strip()
    reference = reference.strip()
    if not candidate or not reference:
        raise ValueError("candidate and reference instructions must be non-empty")
    payload = json.dumps(
        {"candidate": candidate, "reference": reference},
        ensure_ascii=False,
        indent=2,
    )
    user = f"Score this instruction pair from 1 to 5:\n{payload}"
    if validation_error:
        user += (
            "\n\nYour previous response failed deterministic validation: "
            f"{validation_error}. Return a corrected JSON judgment."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def validate_judgment(
    value: Any,
    *,
    candidate: str | None = None,
    reference: str | None = None,
) -> int:
    if not isinstance(value, Mapping):
        raise ValueError("judgment must be an object")
    if set(value) != {"score", "reason"}:
        raise ValueError("judgment must contain exactly score and reason")
    score = value["score"]
    reason = value["reason"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("score must be an integer from 1 to 5")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        raise ValueError("reason must be a non-empty string of at most 240 characters")
    if candidate is not None and reference is not None and candidate.strip() == reference.strip():
        if score != 5:
            raise ValueError("identical instructions must receive score 5")
    return score


def parse_judgment_text(
    text: str,
    *,
    candidate: str,
    reference: str,
) -> tuple[dict[str, Any], int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("judge returned an empty response")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge returned invalid JSON: {exc}") from exc
    score = validate_judgment(value, candidate=candidate, reference=reference)
    return value, score


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class CachedQwenInstructionJudge:
    """Load and validate a complete pinned Qwen judgment JSONL cache."""

    def __init__(
        self,
        path: Path,
        expected_pairs: Sequence[tuple[str, str]],
        *,
        model_name: str = QWEN_JUDGE_MODEL_NAME,
        model_revision: str = QWEN_JUDGE_MODEL_REVISION,
    ) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise ValueError(f"Qwen judgment cache is not a file: {self.path}")
        scores: dict[tuple[str, str], int] = {}
        pair_ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_number}: invalid JSON") from exc
                if not isinstance(record, Mapping):
                    raise ValueError(f"{self.path}:{line_number}: record must be an object")
                candidate = record.get("candidate")
                reference = record.get("reference")
                if not isinstance(candidate, str) or not isinstance(reference, str):
                    raise ValueError(f"{self.path}:{line_number}: invalid instruction pair")
                candidate = candidate.strip()
                reference = reference.strip()
                expected_id = instruction_pair_id(candidate, reference)
                if record.get("pair_id") != expected_id or expected_id in pair_ids:
                    raise ValueError(f"{self.path}:{line_number}: invalid or duplicate pair_id")
                if record.get("judge_version") != QWEN_JUDGE_VERSION:
                    raise ValueError(f"{self.path}:{line_number}: unsupported judge_version")
                model = record.get("model")
                if not isinstance(model, Mapping) or model.get("name") != model_name:
                    raise ValueError(f"{self.path}:{line_number}: wrong judge model")
                if model.get("revision") != model_revision:
                    raise ValueError(f"{self.path}:{line_number}: wrong judge model revision")
                score = validate_judgment(
                    record.get("judgment"),
                    candidate=candidate,
                    reference=reference,
                )
                pair_ids.add(expected_id)
                scores[(candidate, reference)] = score

        expected = {(candidate.strip(), reference.strip()) for candidate, reference in expected_pairs}
        actual = set(scores)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(
                "Qwen judgment cache pair coverage mismatch: "
                f"missing={len(missing)} extra={len(extra)}"
            )
        values = list(scores.values())
        self._scores = scores
        self.metadata: Mapping[str, Any] = {
            "metric": "Qwen3.5-9B semantic instruction score (1--5)",
            "judge": model_name,
            "model_revision": model_revision,
            "judge_version": QWEN_JUDGE_VERSION,
            "prompt_version": QWEN_PROMPT_VERSION,
            "judgment_cache_path": str(self.path),
            "judgment_cache_sha256": _file_sha256(self.path),
            "unique_instruction_pairs": len(scores),
            "pair_score_min": min(values) if values else None,
            "pair_score_max": max(values) if values else None,
        }

    def __call__(self, candidate: str, reference: str) -> int:
        key = (candidate.strip(), reference.strip())
        try:
            return self._scores[key]
        except KeyError as exc:
            raise KeyError(f"instruction pair was not precomputed: {key!r}") from exc
