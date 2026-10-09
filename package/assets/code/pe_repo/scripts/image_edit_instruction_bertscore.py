"""Pinned BERTScore-F1 support for Hard1000 instruction evaluation."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


BERT_SCORE_DISTRIBUTION_VERSION = "0.3.13"
BERT_SCORE_BASELINE_RELATIVE_PATH = (
    "rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv"
)
BERT_SCORE_BASELINE_SHA256 = (
    "e9ea1c46dddcf57e19f6e6a57eabcfc38eefa85d4377b39e6b810741be9c2129"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(root: Path) -> str:
    """Hash relative paths, sizes, and bytes for every regular model file."""

    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"local model directory contains no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


class CachedBERTScoreF1:
    """Compute official baseline-rescaled BERTScore-F1 once per unique pair."""

    def __init__(
        self,
        model_path: Path,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_length: int,
        device: str,
        model_name: str,
        model_revision: str,
        num_layers: int,
        expected_model_sha256: str,
    ) -> None:
        if batch_size <= 0 or max_length <= 0 or num_layers <= 0:
            raise ValueError("batch_size, max_length, and num_layers must be positive")
        self.model_path = model_path.expanduser().resolve()
        if not self.model_path.is_dir():
            raise ValueError(f"local model path is not a directory: {self.model_path}")
        actual_model_sha256 = directory_sha256(self.model_path)
        if actual_model_sha256 != expected_model_sha256:
            raise ValueError(
                "BERTScore model SHA-256 mismatch: "
                f"expected={expected_model_sha256} actual={actual_model_sha256}"
            )

        package_version = importlib_metadata.version("bert-score")
        if package_version != BERT_SCORE_DISTRIBUTION_VERSION:
            raise ValueError(
                "unsupported bert-score distribution: "
                f"expected={BERT_SCORE_DISTRIBUTION_VERSION} actual={package_version}"
            )

        import torch
        import transformers
        import bert_score
        from bert_score import BERTScorer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but unavailable: {device}")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)

        baseline_path = (
            Path(bert_score.__file__).resolve().parent / BERT_SCORE_BASELINE_RELATIVE_PATH
        )
        if not baseline_path.is_file():
            raise ValueError(f"BERTScore baseline file is missing: {baseline_path}")
        actual_baseline_sha256 = file_sha256(baseline_path)
        if actual_baseline_sha256 != BERT_SCORE_BASELINE_SHA256:
            raise ValueError(
                "BERTScore baseline SHA-256 mismatch: "
                f"expected={BERT_SCORE_BASELINE_SHA256} actual={actual_baseline_sha256}"
            )

        unique_pairs = sorted(
            {
                (candidate.strip(), reference.strip())
                for candidate, reference in pairs
                if candidate.strip() and reference.strip()
            }
        )
        if len(unique_pairs) != len(
            {
                (candidate.strip(), reference.strip())
                for candidate, reference in pairs
            }
        ):
            raise ValueError("BERTScore instruction pairs must contain non-empty text")

        scorer = BERTScorer(
            model_type=str(self.model_path),
            num_layers=num_layers,
            batch_size=batch_size,
            idf=False,
            device=device,
            lang="en",
            rescale_with_baseline=True,
            baseline_path=str(baseline_path),
            use_fast_tokenizer=False,
        )
        # DeBERTa's legacy tokenizer config has no finite model_max_length. The
        # official scorer reads this value, so freeze it explicitly for modern
        # Transformers versions and reproducible truncation behavior.
        scorer._tokenizer.model_max_length = max_length

        candidates = [candidate for candidate, _ in unique_pairs]
        references = [reference for _, reference in unique_pairs]
        if unique_pairs:
            _, _, f1 = scorer.score(
                candidates,
                references,
                batch_size=batch_size,
                verbose=True,
            )
            values = [float(value) for value in f1.tolist()]
        else:
            values = []
        if len(values) != len(unique_pairs) or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError("official BERTScore returned invalid F1 values")

        self._scores = dict(zip(unique_pairs, values, strict=True))
        self.metadata: Mapping[str, Any] = {
            "metric": "BERTScore-F1",
            "implementation": "bert-score",
            "distribution_version": package_version,
            "internal_version": bert_score.__version__,
            "configuration_hash": scorer.hash,
            "model": model_name,
            "model_revision": model_revision,
            "model_path": str(self.model_path),
            "model_sha256": actual_model_sha256,
            "num_layers": num_layers,
            "idf": False,
            "language": "en",
            "rescale_with_baseline": True,
            "baseline_path": BERT_SCORE_BASELINE_RELATIVE_PATH,
            "baseline_sha256": actual_baseline_sha256,
            "use_fast_tokenizer": False,
            "max_length": max_length,
            "batch_size": batch_size,
            "device": device,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "deterministic_algorithms": True,
            "tf32": False,
            "unique_instruction_pairs": len(unique_pairs),
            "pair_score_min": min(values) if values else None,
            "pair_score_max": max(values) if values else None,
        }

    def __call__(self, candidate: str, reference: str) -> float:
        key = (candidate.strip(), reference.strip())
        try:
            return self._scores[key]
        except KeyError as exc:
            raise KeyError(f"instruction pair was not precomputed: {key!r}") from exc
