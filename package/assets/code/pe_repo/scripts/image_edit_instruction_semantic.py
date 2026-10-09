"""Pinned, local-only instruction embedding support for Hard1000 evaluation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class TextEncoder(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


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


class HuggingFaceMeanPoolEncoder:
    """Encode text using the benchmark's pinned mean-pooling contract."""

    def __init__(
        self,
        model_path: Path,
        *,
        batch_size: int,
        max_length: int,
        device: str,
        model_name: str,
        model_revision: str,
        expected_sha256: str,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self.model_path = model_path.expanduser().resolve()
        if not self.model_path.is_dir():
            raise ValueError(f"local model path is not a directory: {self.model_path}")
        actual_sha256 = directory_sha256(self.model_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"semantic model SHA-256 mismatch: expected={expected_sha256} actual={actual_sha256}"
            )

        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but unavailable: {device}")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        )
        self.model.to(self.device)
        self.model.eval()
        self._metadata = {
            "backend": "transformers.AutoModel",
            "local_files_only": True,
            "path": str(self.model_path),
            "name": model_name,
            "revision": model_revision,
            "sha256": actual_sha256,
            "pooling": "attention-mask mean pooling over last_hidden_state",
            "embedding_normalization": "L2",
            "similarity": "cosine clipped to [0,1]",
            "device": str(self.device),
            "dtype": str(getattr(self.model, "dtype", "unknown")),
            "batch_size": batch_size,
            "max_length": max_length,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        torch = self.torch
        vectors: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    raise ValueError("encoder output has no last_hidden_state")
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())
        return vectors


class CachedSemanticSimilarity:
    """Cosine similarity over one deterministic, pre-batched embedding table."""

    def __init__(self, encoder: TextEncoder, texts: Sequence[str]) -> None:
        unique = sorted({text.strip() for text in texts if text.strip()})
        vectors = encoder.encode(unique) if unique else []
        if len(vectors) != len(unique):
            raise ValueError(f"encoder returned {len(vectors)} vectors for {len(unique)} texts")
        self.metadata = dict(encoder.metadata)
        self.metadata["encoded_unique_nonempty_texts"] = len(unique)
        self._vectors: dict[str, tuple[float, ...]] = {}
        dimension: int | None = None
        for text, vector in zip(unique, vectors, strict=True):
            values = tuple(float(value) for value in vector)
            if not values or any(not math.isfinite(value) for value in values):
                raise ValueError(f"encoder returned an invalid vector for {text!r}")
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                raise ValueError("encoder returned inconsistent embedding dimensions")
            norm = math.sqrt(sum(value * value for value in values))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ValueError(f"encoder returned a zero-norm vector for {text!r}")
            self._vectors[text] = tuple(value / norm for value in values)
        self.metadata["embedding_dimension"] = dimension

    def __call__(self, left: str, right: str) -> float:
        left = left.strip()
        right = right.strip()
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        cosine = sum(
            a * b for a, b in zip(self._vectors[left], self._vectors[right], strict=True)
        )
        return max(0.0, min(1.0, float(cosine)))
