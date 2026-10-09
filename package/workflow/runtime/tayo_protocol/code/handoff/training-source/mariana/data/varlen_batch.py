"""Storage-independent variable-length token-budget packing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
import torch.nn.functional as F


def patchify_image(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert a patch-aligned ``[C, H, W]`` image to flat patch rows."""
    if image.ndim != 3:
        raise ValueError(f"image must be [C, H, W], got {tuple(image.shape)}")
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    height, width = (int(value) for value in image.shape[-2:])
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image {width}x{height} is not aligned to patch_size={patch_size}"
        )
    return (
        F.unfold(image.unsqueeze(0), kernel_size=patch_size, stride=patch_size)
        .transpose(1, 2)
        .squeeze(0)
    )


@dataclass
class PackedVarlenBatch:
    """One fixed-token-budget pack of variable-resolution samples.

    Per sample the model sequence is ``[text rows, image rows]`` with
    ``seqlen_i = text_lens[i] + img_lens[i]``. ``text_tokens=None`` selects
    online text encoding from ``captions`` and per-sample ``metadata``.

    Geometry and length tensors deliberately remain on the host when
    :meth:`to` is called; PixelCraft's varlen trainer consumes them as host
    control data and creates only the required device tensors.
    """

    img_patches: torch.Tensor
    img_grids: torch.Tensor
    img_lens: torch.Tensor
    text_tokens: torch.Tensor | None
    text_lens: torch.Tensor
    cu_seqlens: torch.Tensor
    pack_target: int
    pad_tokens: int
    max_seqlen: int
    repa_patch_mask: torch.Tensor | None
    captions: list[str]
    metadata: list[dict[str, Any]]
    packer_stats: dict[str, float] | None = None
    is_padding: bool = False

    @property
    def n_samples(self) -> int:
        return int(self.img_lens.shape[0])

    @property
    def total_tokens(self) -> int:
        return int(self.cu_seqlens[-1].item())

    def to(self, device=None, **kwargs: Any) -> "PackedVarlenBatch":
        non_blocking = bool(kwargs.get("non_blocking", self.img_patches.is_pinned()))
        self.img_patches = self.img_patches.to(device, non_blocking=non_blocking)
        if self.text_tokens is not None:
            self.text_tokens = self.text_tokens.to(device, non_blocking=non_blocking)
        if self.repa_patch_mask is not None:
            self.repa_patch_mask = self.repa_patch_mask.to(
                device, non_blocking=non_blocking
            )
        return self

    def pin_memory(self) -> "PackedVarlenBatch":
        self.img_patches = self.img_patches.pin_memory()
        self.img_grids = self.img_grids.pin_memory()
        self.img_lens = self.img_lens.pin_memory()
        if self.text_tokens is not None:
            self.text_tokens = self.text_tokens.pin_memory()
        self.text_lens = self.text_lens.pin_memory()
        self.cu_seqlens = self.cu_seqlens.pin_memory()
        if self.repa_patch_mask is not None:
            self.repa_patch_mask = self.repa_patch_mask.pin_memory()
        return self


T = TypeVar("T")


@dataclass(slots=True)
class TokenBudgetItem(Generic[T]):
    cost: int
    nbytes: int
    text_len: int
    image_len: int
    grid_h: int
    grid_w: int
    sample: T


class TokenBudgetPacker(Generic[T]):
    """Arrival-order greedy packer with bounded lookahead.

    Intake is bounded by sample count and bytes. Emission scans oldest-first
    and takes every item that fits the remaining token budget. Items that do
    not fit remain in arrival order for the next pack.
    """

    _STAT_KEYS = (
        "packs",
        "samples",
        "tokens",
        "pad_tokens",
        "buf_samples_hwm",
        "buf_bytes_hwm",
    )

    def __init__(
        self,
        target_T: int,
        max_samples: int,
        max_bytes: int,
        max_samples_per_pack: int | None = None,
    ) -> None:
        self.target_T = int(target_T)
        self.max_samples = int(max_samples)
        self.max_bytes = int(max_bytes)
        self.max_samples_per_pack = (
            int(max_samples_per_pack) if max_samples_per_pack is not None else None
        )
        if self.target_T <= 0:
            raise ValueError(f"target_T must be positive, got {self.target_T}")
        if self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {self.max_samples}")
        if self.max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {self.max_bytes}")
        if self.max_samples_per_pack is not None and self.max_samples_per_pack <= 0:
            raise ValueError(
                "max_samples_per_pack must be positive when set, got "
                f"{self.max_samples_per_pack}"
            )
        self._buf: list[TokenBudgetItem[T]] = []
        self._buf_bytes = 0
        self.stats = {key: 0 for key in self._STAT_KEYS}

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def buffered_bytes(self) -> int:
        return self._buf_bytes

    @property
    def buffered_items(self) -> tuple[TokenBudgetItem[T], ...]:
        """Read-only view used to build byte-free resume descriptors."""
        return tuple(self._buf)

    def wants_more(self) -> bool:
        if not self._buf:
            return True
        return len(self._buf) < self.max_samples and self._buf_bytes < self.max_bytes

    def _validate_item(self, item: TokenBudgetItem[T]) -> None:
        if int(item.cost) <= 0:
            raise ValueError(f"item cost must be positive, got {item.cost}")
        if int(item.cost) > self.target_T:
            raise ValueError(
                f"TokenBudgetPacker.add: item cost {item.cost} exceeds "
                f"target_T {self.target_T}"
            )
        if int(item.nbytes) < 0:
            raise ValueError(f"item nbytes must be non-negative, got {item.nbytes}")

    def add(self, item: TokenBudgetItem[T]) -> None:
        self._validate_item(item)
        self._buf.append(item)
        self._buf_bytes += int(item.nbytes)
        self.stats["buf_samples_hwm"] = max(
            self.stats["buf_samples_hwm"], len(self._buf)
        )
        self.stats["buf_bytes_hwm"] = max(
            self.stats["buf_bytes_hwm"], self._buf_bytes
        )

    def restore_buffer(self, items: list[TokenBudgetItem[T]]) -> None:
        """Restore replayed items without advancing selection statistics."""
        for item in items:
            self._validate_item(item)
        self._buf = list(items)
        self._buf_bytes = sum(int(item.nbytes) for item in items)

    def emit_pack(self) -> tuple[list[TokenBudgetItem[T]], int]:
        budget = self.target_T
        pack: list[TokenBudgetItem[T]] = []
        leftover: list[TokenBudgetItem[T]] = []
        cap = self.max_samples_per_pack
        for item in self._buf:
            if item.cost <= budget and (cap is None or len(pack) < cap):
                pack.append(item)
                budget -= item.cost
            else:
                leftover.append(item)
        self._buf = leftover
        if pack:
            self._buf_bytes -= sum(int(item.nbytes) for item in pack)
            self.stats["packs"] += 1
            self.stats["samples"] += len(pack)
            self.stats["tokens"] += self.target_T - budget
            self.stats["pad_tokens"] += budget
        return pack, budget

    def config_state_dict(self) -> dict[str, int | None]:
        return {
            "target_T": self.target_T,
            "max_samples": self.max_samples,
            "max_bytes": self.max_bytes,
            "max_samples_per_pack": self.max_samples_per_pack,
        }

    def state_dict(self, *, include_buffer: bool = True) -> dict[str, Any]:
        state: dict[str, Any] = {
            **self.config_state_dict(),
            "stats": dict(self.stats),
        }
        if include_buffer:
            state["buf"] = list(self._buf)
            state["buf_bytes"] = self._buf_bytes
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            self.restore_buffer([])
            return
        for key, expected in self.config_state_dict().items():
            if key not in state:
                continue
            got = state[key]
            if got is not None:
                got = int(got)
            if got != expected:
                raise ValueError(
                    f"TokenBudgetPacker.load_state_dict {key} mismatch: "
                    f"got {got}, expected {expected}"
                )
        items = list(state.get("buf", []))
        self.restore_buffer(items)
        if "buf_bytes" in state and int(state["buf_bytes"]) != self._buf_bytes:
            raise ValueError(
                "TokenBudgetPacker.load_state_dict buf_bytes mismatch: "
                f"got {state['buf_bytes']}, recomputed {self._buf_bytes}"
            )
        loaded_stats = dict(state.get("stats", {}))
        self.stats = {
            key: int(loaded_stats.get(key, self.stats.get(key, 0)))
            for key in self._STAT_KEYS
        }


__all__ = [
    "PackedVarlenBatch",
    "TokenBudgetItem",
    "TokenBudgetPacker",
    "patchify_image",
]
