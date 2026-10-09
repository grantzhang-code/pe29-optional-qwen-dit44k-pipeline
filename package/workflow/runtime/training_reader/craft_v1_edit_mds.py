"""Native editing MDS transport using shared edit semantics and Mariana resume.

The stored images/bboxes are final. Validate their declared geometry; do not
replay offline resize/crop metadata or fetch original media. ROI-VL extraction
is the existing V7-1 model conditioning operation and remains unchanged.
"""
from __future__ import annotations

import copy
import io
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

from PIL import Image

from mariana.data.dataset.mds_dataset import MdsStreamsConfig
from mariana.data.dataset.pipeline_dataset import PipelineMdsDataset
from mariana.data.dataset.pipeline_varlen import PipelineVarlenProcessor, ReplayKey, semantic_digest
from mariana.data.processors.base import Keys, run_pipeline
from src.data.dataset.craft_v1_image_edit_caption import BASELINE_REFERENCE_REGION_VARIANTS
from src.data.dataset.craft_v1_edit_mds_annotation import NativeEditAnnotation
from src.data.dataset.craft_v1_edit_media import decode_reference_images, edit_media_digest
from src.data.dataset.craft_v1_image_edit_packing import CraftV1EditSampleBuilder, _PreparedEditSample
from src.data.dataset.parquet_bucket_dataset import _LightSample, _build_meta_filter


class CraftV1EditMdsDataset(PipelineMdsDataset):
    """Mosaic storage plus the same flat edit configuration as the Parquet path.

    Packing parameters have one owner, the existing dataset overrides. The
    datamodule's varlen settings must interpolate those same values; the
    processor verifies them rather than silently overriding either copy.
    """

    def __init__(
        self, streams: MdsStreamsConfig, *, seed: int = 42, shuffle: bool = True,
        raw_batch_size: int = 32, streaming_kwargs: dict[str, Any] | None = None,
        meta_filters: list[dict[str, Any]] | None = None,
        **edit_kwargs: Any,
    ) -> None:
        # Share Parquet's declarative predicates, but require configured fields:
        # a missing source label must never silently admit another dataset.
        self.meta_filter_fn = _build_meta_filter(meta_filters, strict=True)
        self.edit = CraftV1EditSampleBuilder(seed=seed, **edit_kwargs)
        if self.edit.variant not in BASELINE_REFERENCE_REGION_VARIANTS:
            raise ValueError("native editing MDS supports v7+v4-baseline, v7+v4-refid and v7+v4-grouped")
        if self.edit.reference_image_columns != ["input_images"]:
            raise ValueError("native MDS reference image column is input_images")
        super().__init__(
            streams, [NativeEditAnnotation(self.edit.annotation_column)], seed=seed, shuffle=shuffle,
            raw_batch_size=raw_batch_size, streaming_kwargs=streaming_kwargs,
        )

    @property
    def patch_size(self) -> int:
        return self.edit.pixel_patch_size

    def configure_from_denoiser(self, denoiser: Any) -> None:
        self.edit.configure_from_denoiser(denoiser)

    def configure_text_tokenizer_from_conditioner(self, conditioner: Any) -> None:
        self.edit.configure_text_tokenizer_from_conditioner(conditioner)
        self.text_budget = self.edit._text_budget

    def should_skip_sample(self, sample: dict[str, Any]) -> bool:
        if self.meta_filter_fn is not None and not self.meta_filter_fn(sample):
            return True
        return not self.edit.accept_row(sample)

    def build_varlen_processor(self, **packing_config: Any):
        return CraftV1EditMdsProcessor(self, **packing_config)

    def __iter__(self):
        raise RuntimeError("editing MDS requires the datamodule's resumable varlen loader")


@dataclass(slots=True)
class _EditMdsSample:
    prepared: _PreparedEditSample
    replay_key: ReplayKey
    media_digest: str
    semantic_digest: str | None = None


class CraftV1EditMdsProcessor(PipelineVarlenProcessor):
    """Specialize only sample preparation and multi-image assembly.

    The bounded lookahead retains prepared tensors, as the existing editing
    packer does. Their image/ROI bytes are included in buffer admission.
    Checkpoints remain byte-free: Mosaic ids + verified replay descriptors.
    """

    def __init__(self, dataset: CraftV1EditMdsDataset, **packing_config: Any) -> None:
        super().__init__(dataset, **packing_config)
        self.edit = dataset.edit
        for name in ("pack_target_T", "pack_buffer_size", "pack_buffer_bytes", "pack_max_samples_per_pack"):
            if getattr(self, name) != getattr(self.edit, name):
                raise ValueError(f"editing MDS {name} must match the dataset's packing setting")
        self._pack_skip.update({"edit_schema": 0, "stored_geometry": 0, "text_admission": 0})

    def _stored_image(
        self, value: Any, *, name: str, geometry: Mapping | None = None,
    ) -> tuple[bytes, int, int]:
        if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
            raise ValueError(f"{name}: missing encoded image bytes")
        encoded = bytes(value)
        with Image.open(io.BytesIO(encoded)) as image:
            width, height = image.size
        patch = self.edit.pixel_patch_size
        if width % patch or height % patch:
            raise ValueError(f"{name}: stored size {width}x{height} violates patch/area contract; refusing to resize")
        # Enforce final stored planes, not the producer's resize algorithm.
        # Nearest-patch releases may exceed the continuous area cap by half
        # a patch per edge. Bound that allowance without requiring provenance.
        if width * height > self.edit.image_max_pixels:
            lower_area = max(1, width - patch / 2) * max(1, height - patch / 2)
            if lower_area > self.edit.image_max_pixels:
                raise ValueError(f"{name}: stored size {width}x{height} violates patch/area contract; refusing to resize")
        if geometry is not None:
            for key, actual in (("final_width", width), ("final_height", height)):
                declared = geometry.get(key)
                if declared is not None and (type(declared) is not int or declared != actual):
                    raise ValueError(f"{name}: encoded dimensions disagree with stored geometry")
        return encoded, width, height

    def _stored_resize_contract(self, row: Mapping) -> Mapping | None:
        """Normalize optional producer metadata; validate final-plane conflicts.

        Versions and original resize plans are provenance, not a reader ABI.
        Absent geometry is supported; supplied final dimensions and reference
        ordering must agree with the encoded media checked by _stored_image.
        """
        metadata = row.get("meta_info") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, Mapping):
            raise ValueError("meta_info must be an object")
        contract = metadata.get("mds_resize_contract")
        if contract is not None and not isinstance(contract, Mapping):
            raise ValueError("mds_resize_contract must be an object")
        for declaration in (metadata, contract or {}):
            for key in ("bbox_order", "bbox_output_order", "annotation_bbox_output_order"):
                if declaration.get(key) not in (None, "yxyx_1000"):
                    raise ValueError(f"{key}: stored annotation must use yxyx_1000")

        def normalize(value):
            if isinstance(value, str):
                value = json.loads(value)
            if value is None or value == {}:
                return None
            if not isinstance(value, Mapping):
                raise ValueError("stored geometry must be an object")
            sources = value.get("sources")
            if sources is None and "source" in value:
                sources = [value["source"]]
            if sources is not None and (not isinstance(sources, list) or not sources
                                        or any(not isinstance(g, Mapping) for g in sources)):
                raise ValueError("stored geometry requires ordered reference planes")
            target = value.get("target")
            if target is not None and not isinstance(target, Mapping):
                raise ValueError("stored target geometry must be an object")
            if sources is None and target is None:
                return None
            return {"sources": sources, "target": target}

        declarations = [normalize(contract), normalize(metadata.get("geometry")),
                        normalize(row.get("resize_geometry"))]
        merged = {"sources": None, "target": None}
        for geometry in declarations:
            if geometry is None:
                continue
            for side in ("sources", "target"):
                incoming = geometry[side]
                if incoming is None:
                    continue
                if merged[side] is None:
                    merged[side] = [dict(g) for g in incoming] if side == "sources" else dict(incoming)
                    continue
                old = merged[side] if side == "sources" else [merged[side]]
                new = incoming if side == "sources" else [incoming]
                if len(old) != len(new):
                    raise ValueError("conflicting stored reference counts")
                for before, after in zip(old, new):
                    for key in ("final_width", "final_height"):
                        if key in after:
                            if key in before and before[key] != after[key]:
                                raise ValueError("conflicting stored final dimensions")
                            before[key] = after[key]
        return merged if any(v is not None for v in merged.values()) else None

    def _to_light_sample(self, raw, *, epoch=None, count_drops=True) -> _EditMdsSample | None:
        # Replay calls this directly, bypassing feed_raw_sample's admission.
        if self.dataset.meta_filter_fn is not None and not self.dataset.meta_filter_fn(raw):
            return None
        if epoch is None:
            epoch = self.current_epoch()
        key = self.replay_key_for_raw(raw, epoch=epoch)
        # A Mosaic index must never replace the stable original editing identity.
        if raw.get("sample_id") is None or not str(raw["sample_id"]):
            raise ValueError("editing MDS requires the original sample_id")
        try:
            row = run_pipeline(dict(raw), self.dataset.pipeline,
                               self._ctx(sample_id=key.sample_id, epoch=key.epoch))
        except Exception:
            if count_drops:
                self._pack_skip["edit_schema"] += 1
            raise
        if row is None:
            return None
        try:
            resize = self._stored_resize_contract(row)
            target, width, height = self._stored_image(
                row.get("output_image"), name="output_image",
                geometry=resize["target"] if resize is not None else None,
            )
            reference_bytes = decode_reference_images(row, max_images=self.edit.max_reference_images)
            if resize is not None and resize["sources"] is not None and len(resize["sources"]) != len(reference_bytes):
                raise ValueError("resize contract must describe every reference image in order")
            sources = [self._stored_image(value, name=f"input_images[{i}]",
                                         geometry=resize["sources"][i] if resize is not None and resize["sources"] is not None else None)
                       for i, value in enumerate(reference_bytes)]
            count = row.get("num_inputs")
            if count is not None and (isinstance(count, bool) or not isinstance(count, Integral)
                                      or count != len(sources)):
                raise ValueError("num_inputs must match the ordered reference images")
            for name, actual in (("input_widths", [w for _, w, _ in sources]),
                                 ("input_heights", [h for _, _, h in sources])):
                declared = row.get(name)
                if isinstance(declared, str):
                    declared = json.loads(declared)
                if declared is not None and declared != actual:
                    raise ValueError(f"{name}: encoded dimensions disagree with metadata")
            paths = row.get("input_s3_paths")
            # Video-derived rows embed their images without original object
            # URIs. A supplied nonempty path list must still match frame order.
            if isinstance(paths, list) and paths and len(paths) != len(sources):
                raise ValueError("input_s3_paths must match the ordered reference images")
            row["input_images"] = reference_bytes
            for name, actual in (("output_width", width), ("output_height", height)):
                if row.get(name) is not None and int(row[name]) != actual:
                    raise ValueError(f"{name}: encoded dimensions disagree with metadata")
        except Exception:
            if count_drops:
                self._pack_skip["stored_geometry"] += 1
            raise
        builder = self.edit
        stored_area = max([width * height, *(w * h for _, w, h in sources)])
        if stored_area > builder.image_max_pixels:
            # Validation above authorizes only the producer's exact rounding.
            # Keep that already-final image plane intact throughout caption,
            # ROI and position preparation. Never mutate the shared builder;
            # the independent ROI allocation and packed-token caps stay fixed.
            builder = copy.copy(builder)
            builder.image_max_pixels = stored_area
        source_path = str(row.get("output_s3_path") or "mds")
        light = _LightSample(
            bin_name=f"native_{width}x{height}", image_bytes=target,
            orig_w=width, orig_h=height, target_w=width, target_h=height, caption="",
            metadata={"sample_id": str(row["sample_id"]), Keys.SAMPLE_ID: key.sample_id,
                      "sample_key": str(row["sample_id"]), "source_path": source_path,
                      "data_epoch": key.epoch, "is_rgba": False},
        )
        light = builder.prepare_row(row, light, source_path=source_path)
        if light is None:
            return None
        final = builder._finalize_sample(light)
        if ((final.target.final_width, final.target.final_height) != (width, height)
                or len(final.references) != len(sources)
                or any((plane.final_width, plane.final_height) != (width, height)
                       for plane, (_, width, height) in zip(final.references, sources))):
            raise RuntimeError("shared editing preparation unexpectedly changed stored MDS geometry")
        if not builder.json_post_encoder_dropout:
            from src.data.dataset.craft_v1_edit_clean_caption import clean_inference_caption

            # Disabling training randomness must retain its deterministic
            # leaf-only palettes, canonical JSON and description capitalization.
            # Do this before tokenization, position spans and ROI expansion.
            final = replace(final,
                raw_caption=clean_inference_caption(final.raw_caption),
                eligibility_caption=clean_inference_caption(final.eligibility_caption))
        prepared = builder._prepare_text(final)
        if prepared is None:
            if count_drops:
                self._pack_skip["text_admission"] += 1
            return None
        item = self.edit._packer_item(prepared)
        if item is None:
            if count_drops:
                self._pack_skip["oversize"] += 1
            return None
        media_digest = edit_media_digest(target, reference_bytes)
        return _EditMdsSample(prepared, replace(
            key, cost=item.cost, nbytes=item.nbytes, text_len=item.text_len,
            image_len=item.image_len, grid_h=item.grid_h, grid_w=item.grid_w,
        ), media_digest)

    def replay_key_for_item(self, item) -> ReplayKey:
        sample = item.sample
        if sample.semantic_digest is None:
            sample.semantic_digest = semantic_digest({
                "caption": sample.prepared.caption, "metadata": sample.prepared.metadata,
                "media": sample.media_digest,
            })
        return replace(sample.replay_key, digest=sample.semantic_digest)

    def _decode_pack(self, items, pad_tokens):
        started = time.perf_counter()
        edit_items = [replace(item, sample=item.sample.prepared) for item in items]
        batch = self.edit._build_batch(edit_items, pad_tokens, self.packer)
        return batch, (time.perf_counter() - started) * 1000, 0, 0, 0

    def _finalize_pack(self, batch, **kwargs):
        edit_stats = dict(batch.packer_stats) if batch is not None else {}
        batch = super()._finalize_pack(batch, **kwargs)
        if batch is not None:
            # Preserve editing-specific diagnostics alongside standard I/O timing.
            batch.packer_stats = {**edit_stats, **batch.packer_stats,
                **{f"{name}_dropped": float(self._pack_skip[name])
                   for name in ("edit_schema", "stored_geometry", "text_admission")}}
        return batch
