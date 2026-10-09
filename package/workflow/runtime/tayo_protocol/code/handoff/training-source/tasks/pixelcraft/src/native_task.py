"""PixelCraft training task for Mariana's native runner."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from mariana.core.metrics import debug_metrics_enabled
from mariana.core.runtime import RuntimeContext
from mariana.core.cuda_stream import record_cuda_tensors
from mariana.core.step import StepOutput, StepState
from mariana.core.task import OptimizerParameters
from mariana.core.phase_timing import begin_step, is_enabled, record_event
from mariana.diffusion import DiffusionTask
from mariana.integrations.transformer_engine import (
    apply_te_fp8,
    fp8_autocast_ctx,
    is_te_fp8_enabled,
    plan_te_fp8_replacement,
    resolve_te_integration_config,
    should_replace_conditioner,
    should_replace_ema,
    should_replace_main_network,
    validate_te_replacement,
)
from mariana.perf.native_parallel import NativeParallel, ParallelConfig
from mariana.integrations import flash_attention_4
from src.models.autoencoder.base import fp2uint8
from src.utils import compile_cache
from src.models.layers import attention_op
from src.models.layers.fused_ops import prewarm_fused_rmsnorm_modulate
from src.utils.compile_cache import is_compile_disabled
from src.utils.copy import copy_params
from src.utils.no_grad import no_grad
from src.utils.shm_cache import shm_cleanup_barrier


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class PixelCraftNativeTask(DiffusionTask):
    """Own PixelCraft modules and expose one native ``StepOutput`` forward."""

    _EXPLICIT_MUON_NAMES = (
        "shared_adaln_modulation.proj.weight",
        "final_layer.adaLN_modulation.proj.weight",
        "t_embedder.mlp.2.weight",
    )

    def __init__(
        self,
        *,
        vae: nn.Module,
        conditioner: nn.Module,
        denoiser: nn.Module,
        diffusion_trainer: nn.Module,
        diffusion_sampler: nn.Module,
        eval_original_model: bool = False,
        eval_non_ema_alongside_ema: bool = False,
        val_resample_to_input_size: bool = False,
        te_fp8_config: Any = None,
    ) -> None:
        super().__init__(
            vae=vae,
            conditioner=conditioner,
            denoiser=denoiser,
            diffusion_trainer=diffusion_trainer,
            diffusion_sampler=diffusion_sampler,
            eval_original_model=eval_original_model,
            eval_non_ema_alongside_ema=eval_non_ema_alongside_ema,
            val_resample_to_input_size=val_resample_to_input_size,
            sample_postprocess=fp2uint8,
        )
        self.te_fp8_config = (
            resolve_te_integration_config(te_fp8_config)
            if te_fp8_config is not None
            else None
        )
        self._te_fp8_applied = False
        self._te_cond_fp8_applied = False
        self._te_fp8_plan_reports: list[dict[str, Any]] = []
        self._te_fp8_reports: list[dict[str, Any]] = []
        self._te_fp8_validation_reports: list[dict[str, Any]] = []
        self._mup_infshape_cache: Any = None
        self._parallel: NativeParallel | None = None
        self._ctx: RuntimeContext | None = None

    def setup_parallel(
        self,
        cfg: ParallelConfig,
        ctx: RuntimeContext,
    ) -> nn.Module:
        """Freeze task trunks and apply DDP or shared-mesh FSDP2."""

        from mariana.optim.mup_preserve import (
            capture_infshape_cache,
            restore_infshape,
            sync_ema_infshape,
        )

        self._ctx = ctx
        if os.environ.get("PIXELCRAFT_FSDP_CPU_STAGE", "0") == "1":
            return self._setup_parallel_from_cpu(
                cfg=cfg,
                ctx=ctx,
                capture_infshape_cache=capture_infshape_cache,
                restore_infshape=restore_infshape,
                sync_ema_infshape=sync_ema_infshape,
            )

        self.to(ctx.device)
        _barrier()
        shm_cleanup_barrier(_barrier)
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)
        sync_ema_infshape(self.denoiser, self.ema_denoiser)

        no_grad(self.conditioner)
        no_grad(self.vae)
        no_grad(self.ema_denoiser)

        parallel = NativeParallel(cfg, ctx)
        self._parallel = parallel
        self._apply_te_fp8_replacement(
            rank=ctx.rank,
            shard_mesh=parallel.shard_submesh,
        )

        if not cfg.is_sharded:
            self.ema_denoiser.to(torch.float32)
            return parallel.parallelize(self)

        mup_modules = self._mup_modules()
        snapshot = capture_infshape_cache(mup_modules)
        self._mup_infshape_cache = snapshot
        parallel.parallelize(self.denoiser)
        parallel.parallelize(self.ema_denoiser)
        projector = getattr(self.diffusion_trainer, "projector", None)
        if projector is not None:
            parallel.parallelize(projector)
        restore_infshape(snapshot, mup_modules)
        if ctx.is_rank_zero:
            print(
                "[native] FSDP2 wrapped denoiser, EMA denoiser, and "
                "diffusion trainer projector",
                flush=True,
            )
        return self

    def _setup_parallel_from_cpu(
        self,
        *,
        cfg: ParallelConfig,
        ctx: RuntimeContext,
        capture_infshape_cache: Any,
        restore_infshape: Any,
        sync_ema_infshape: Any,
    ) -> nn.Module:
        """Shard the 13B live/EMA trunks before materializing them on CUDA.

        This is an explicit, local-memory startup mode for devices on which
        staging the full FP32 live model, FP32 EMA model, and BF16 27B
        conditioner on every rank would exceed memory before FSDP2 can shard
        them.  It is
        intentionally restricted to the Craft-v1 conditioner-only FP8 recipe:
        replacing either FSDP root after wrapping would invalidate FSDP state.

        The normal production ordering remains the default.  Callers must opt
        in with ``PIXELCRAFT_FSDP_CPU_STAGE=1``.
        """

        te_config = resolve_te_integration_config(self.te_fp8_config)
        violations: list[str] = []
        if not cfg.is_sharded:
            violations.append(
                "parallel strategy is not the validated FSDP2 path or HSDP path"
            )
        if cfg.replicate * cfg.shard != ctx.world_size:
            violations.append(
                "FSDP2 mesh does not cover the distributed world: "
                f"replicate={cfg.replicate} * shard={cfg.shard} != "
                f"world_size={ctx.world_size}"
            )
        if not te_config.enabled:
            violations.append("TE FP8 is disabled")
        if not te_config.replace_conditioner:
            violations.append("conditioner.replace is false")
        if not te_config.conditioner_fp8_weights:
            violations.append("conditioner.fp8_weights is false")
        if te_config.replace_main_network:
            violations.append("main-network TE replacement is enabled")
        if te_config.replace_ema:
            violations.append("EMA TE replacement is enabled")
        if not ctx.cuda_available:
            violations.append("CUDA is unavailable")
        if violations:
            raise RuntimeError(
                "PIXELCRAFT_FSDP_CPU_STAGE=1 only supports sharded, "
                "conditioner-only TE FP8 with FP8 weight storage; "
                + "; ".join(violations)
            )

        cpu_stage_roots = {
            "denoiser": self.denoiser,
            "ema_denoiser": self.ema_denoiser,
            "conditioner": self.conditioner,
            "diffusion_trainer": self.diffusion_trainer,
            "diffusion_sampler": self.diffusion_sampler,
            "vae": self.vae,
        }
        non_cpu: list[str] = []
        for root_name, root in cpu_stage_roots.items():
            for name, value in tuple(root.named_parameters()) + tuple(
                root.named_buffers()
            ):
                if value.device.type != "cpu":
                    non_cpu.append(f"{root_name}.{name}={value.device}")
                    if len(non_cpu) >= 8:
                        break
            if len(non_cpu) >= 8:
                break
        if non_cpu:
            raise RuntimeError(
                "PIXELCRAFT_FSDP_CPU_STAGE=1 requires an all-CPU module "
                f"graph before sharding; offending tensors: {non_cpu}"
            )

        # Every rank must finish constructing its CPU module graph before any
        # rank starts collective FSDP wrapping or rank-local CUDA allocation.
        _barrier()
        shm_cleanup_barrier(_barrier)

        # FSDP2 requires live/EMA values and muP metadata to be synchronized
        # while both roots are still ordinary CPU Parameters.  Never copy the
        # live weights into EMA after wrapping: that can destroy DTensor state.
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)
        sync_ema_infshape(self.denoiser, self.ema_denoiser)
        no_grad(self.conditioner)
        no_grad(self.vae)
        no_grad(self.ema_denoiser)

        parallel = NativeParallel(cfg, ctx)
        self._parallel = parallel
        mup_modules = self._mup_modules()
        snapshot = capture_infshape_cache(mup_modules)
        self._mup_infshape_cache = snapshot

        # These roots are small relative to the denoiser/conditioner.  Moving
        # the trainer before wrapping its projector avoids ever recursing a
        # later ``Module.to`` through a fully-sharded DTensor parameter.
        self.diffusion_trainer.to(ctx.device)
        self.diffusion_sampler.to(ctx.device)

        # ``fully_shard`` accepts the CPU roots and materializes only each
        # rank's local shard on its already-selected CUDA device.
        parallel.parallelize(self.denoiser)
        parallel.parallelize(self.ema_denoiser)
        projector = getattr(self.diffusion_trainer, "projector", None)
        if projector is not None:
            parallel.parallelize(projector)
        restore_infshape(snapshot, mup_modules)

        # Replace the frozen Qwen Linear layers directly from CPU into
        # rank-local FP8 storage, then move only its excluded/nonlinear pieces.
        # The VAE is moved last.  Deliberately do not call ``self.to`` here:
        # it would recurse through the already-wrapped FSDP roots.
        current_device = torch.cuda.current_device()
        if current_device != ctx.local_rank:
            raise RuntimeError(
                "CPU-staged conditioner replacement requires the runner to "
                "select the rank-local CUDA device first; "
                f"current_device={current_device}, local_rank={ctx.local_rank}"
            )
        self._apply_te_fp8_replacement(
            rank=ctx.rank,
            shard_mesh=parallel.shard_submesh,
        )
        self.conditioner.to(ctx.device)
        replaced_count = sum(
            int(report.get("replaced_count", 0))
            for report in self._te_fp8_reports
        )
        if not self._te_cond_fp8_applied or replaced_count <= 0:
            raise RuntimeError(
                "CPU-staged conditioner FP8 replacement produced no TE layers"
            )
        conditioner_violations = [
            f"{name}:device={param.device},requires_grad={param.requires_grad}"
            for name, param in self.conditioner.named_parameters()
            if param.device != ctx.device or param.requires_grad
        ]
        conditioner_violations.extend(
            f"buffer:{name}:device={buffer.device}"
            for name, buffer in self.conditioner.named_buffers()
            if buffer.device != ctx.device
        )
        if conditioner_violations:
            raise RuntimeError(
                "CPU-staged conditioner was not fully materialized as frozen "
                f"rank-local CUDA parameters: {conditioner_violations[:8]}"
            )
        self.vae.to(ctx.device)
        _barrier()

        if ctx.is_rank_zero:
            print(
                "[native] CPU-staged FSDP2 wrapped denoiser, EMA denoiser, "
                "and diffusion trainer projector before CUDA materialization; "
                "conditioner materialized with FP8 weight storage",
                flush=True,
            )
        return self

    @property
    def shard_submesh(self) -> Any | None:
        if self._parallel is None:
            return None
        return self._parallel.shard_submesh

    def optimizer_parameters(self) -> OptimizerParameters:
        return self._build_optimizer_param_groups()

    def prepare_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        if self.shard_submesh is not None and hasattr(optimizer, "attach_shard_mesh"):
            optimizer.attach_shard_mesh(self.shard_submesh)
        return optimizer

    def forward(self, batch: Any, state: StepState) -> StepOutput:
        from src.data.packing.varlen_batch import PackedVarlenBatch

        if isinstance(batch, PackedVarlenBatch):
            output = self._training_step_varlen(batch, state)
            if batch.is_padding:
                output.loss = output.loss * 0.0
                output.metadata["throughput"] = {
                    "samples": 0.0,
                    "total_tokens": 0.0,
                }
            return output
        return super().forward(batch, state)

    def text_bucket_hook(self):
        return self._select_precomputed_text_bucket

    def on_train_start(self, runner: Any) -> None:
        """Native-runner callback for compile-cache warmup and compilation."""

        if is_te_fp8_enabled(self.te_fp8_config) and not self._te_fp8_applied:
            raise RuntimeError(
                "TE FP8 replacement must complete before optimizer setup and "
                "the first forward"
            )

        prewarm_fused_rmsnorm_modulate([self.denoiser, self.ema_denoiser])
        # cache the fa4_varlen_backward_supported outcome before compilation
        flash_attention_4.fa4_varlen_backward_supported()
        # same with lazy import of mariana fa4
        attention_op._get_mariana_fa4()

        if is_compile_disabled():
            return
        from mariana.perf.fsdp2 import compile_blocks

        if runner.ctx.local_rank == 0:
            try:
                compile_cache.download_compile_cache_from_s3()
            except Exception as exc:
                print(
                    f"[compile_cache] warm-start failed (non-fatal): {exc}",
                    flush=True,
                )
        _barrier()
        torch._inductor.config.size_asserts = False

        compiler = compile_cache.CachingCompiler()
        compile_env = {
            "disable_env": "PIXELCRAFT_NO_COMPILE",
            "force_dynamic_env": "PIXELCRAFT_COMPILE_DYNAMIC",
            "reduce_overhead_env": "PIXELCRAFT_COMPILE_REDUCE_OVERHEAD",
            "fullgraph_env": "PIXELCRAFT_COMPILE_FULLGRAPH",
        }
        compile_blocks(
            self.denoiser,
            compile_fn=compiler.compile,
            **compile_env,
        )
        compile_blocks(
            self.ema_denoiser,
            compile_fn=compiler.compile,
            **compile_env,
        )
        if runner.ctx.is_rank_zero:
            compile_cache.log_compile_cache_status()

    def on_train_end(self, runner: Any) -> None:
        if is_compile_disabled() or not runner.ctx.is_rank_zero:
            return
        try:
            compile_cache.sync_compile_cache_to_s3()
        except Exception as exc:
            print(
                f"[compile_cache] final publish failed (non-fatal): {exc}",
                flush=True,
            )

    def _apply_te_fp8_replacement(self, *, rank: int, shard_mesh: Any) -> None:
        te_config = self.te_fp8_config
        if self._te_fp8_applied or not is_te_fp8_enabled(te_config):
            return

        if should_replace_ema(te_config):
            ema_plan = plan_te_fp8_replacement(
                self.ema_denoiser,
                frozen=True,
                config=te_config,
                include_availability=True,
            )
            self._te_fp8_plan_reports.append(ema_plan.to_json())
            ema_report = apply_te_fp8(
                self.ema_denoiser,
                frozen=True,
                rank=rank,
                amax_sync_group=shard_mesh,
                config=te_config,
                return_report=True,
            )
            self._te_fp8_reports.append(ema_report.to_json())
            ema_validation = validate_te_replacement(ema_plan, ema_report)
            self._te_fp8_validation_reports.append(ema_validation.to_json())
            ema_validation.raise_for_error()

        if should_replace_conditioner(te_config):
            resolved = resolve_te_integration_config(te_config)
            cond_tower = getattr(self.conditioner, "model", None)
            if cond_tower is None:
                raise ValueError(
                    "te_fp8_config.conditioner.replace=true but the "
                    f"conditioner {type(self.conditioner).__name__} exposes no "
                    "`.model` HF tower to replace"
                )
            cond_policy = resolved.conditioner_policy
            cond_plan = plan_te_fp8_replacement(
                cond_tower,
                frozen=True,
                policy=cond_policy,
                config=te_config,
                include_availability=True,
            )
            self._te_fp8_plan_reports.append(cond_plan.to_json())
            # amax_sync_group deliberately None: the conditioner's forward
            # runs on the async cond-prefetch thread, and a TE amax
            # all-reduce issued from there can interleave with the main
            # thread's FSDP collectives on the same ranks (silent deadlock).
            # Blackwell's default MXFP8 recipe has no amax history to sync,
            # so the group buys nothing; per-rank scales are also fine for a
            # frozen tower whose inputs differ per rank anyway.
            cond_report = apply_te_fp8(
                cond_tower,
                frozen=True,
                rank=rank,
                amax_sync_group=None,
                policy=cond_policy,
                config=te_config,
                return_report=True,
                fp8_weight_storage=resolved.conditioner_fp8_weights,
            )
            self._te_fp8_reports.append(cond_report.to_json())
            cond_validation = validate_te_replacement(cond_plan, cond_report)
            self._te_fp8_validation_reports.append(cond_validation.to_json())
            cond_validation.raise_for_error()
            # _compute_cond_bundle keys its fp8_autocast on this flag; the
            # cached null embedding is computed inside that same context so
            # CFG's conditional/null pair stays on one numeric path.
            self._te_cond_fp8_applied = True

        if should_replace_main_network(te_config):
            main_plan = plan_te_fp8_replacement(
                self.denoiser,
                frozen=False,
                config=te_config,
                include_availability=True,
            )
            self._te_fp8_plan_reports.append(main_plan.to_json())
            main_report = apply_te_fp8(
                self.denoiser,
                frozen=False,
                rank=rank,
                amax_sync_group=shard_mesh,
                config=te_config,
                return_report=True,
            )
            self._te_fp8_reports.append(main_report.to_json())
            main_validation = validate_te_replacement(main_plan, main_report)
            self._te_fp8_validation_reports.append(main_validation.to_json())
            main_validation.raise_for_error()
        self._te_fp8_applied = True

    def _mup_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        denoiser = getattr(self.denoiser, "module", self.denoiser)
        if getattr(denoiser, "use_mup", False):
            modules["denoiser"] = denoiser
            modules["ema_denoiser"] = getattr(
                self.ema_denoiser,
                "module",
                self.ema_denoiser,
            )
        if getattr(self.diffusion_trainer, "use_mup_projector", False):
            modules["projector"] = self.diffusion_trainer.projector
        return modules

    def _build_optimizer_param_groups(self) -> list[dict[str, Any]]:
        from mariana.optim.param_groups import build_param_groups

        denoiser = getattr(self.denoiser, "module", self.denoiser)
        muon_module_paths: tuple[tuple[str, nn.Module], ...] = ()
        if hasattr(denoiser, "blocks"):
            muon_module_paths = (("denoiser.blocks", denoiser.blocks),)
        log = (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )
        groups = build_param_groups(
            denoiser=denoiser,
            diffusion_trainer=self.diffusion_trainer,
            diffusion_sampler=self.diffusion_sampler,
            muon_module_paths=muon_module_paths,
            explicit_muon_names=self._EXPLICIT_MUON_NAMES,
            disable_muon=os.environ.get("PIXELCRAFT_DISABLE_MUON", "0") == "1",
            infshape_cache=self._mup_infshape_cache,
            sampler_lr_override=1e-3,
            log=log,
        )
        return groups

    @staticmethod
    def _select_precomputed_text_bucket(
        metadata: dict[str, Any],
        global_step: int,
        device: torch.device | None = None,
    ) -> None:
        from mariana.data.text_bucket import select_precomputed_text_bucket
        from src.data.module import _pad_text_embedding_to_length

        select_precomputed_text_bucket(
            metadata,
            global_step,
            pad_fn=_pad_text_embedding_to_length,
            device=device,
        )

    @staticmethod
    def _stack_packed_metadata(metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
        from src.data.module import _stack_metadata

        return _stack_metadata(metadata_list)

    @staticmethod
    def _ragged_input_ids_from_pack(
        metadata_list: list[dict[str, Any]],
    ) -> list[torch.Tensor] | None:
        if not metadata_list:
            return None
        ids_list: list[torch.Tensor] = []
        for metadata in metadata_list:
            ids = metadata.get("qwen_input_ids")
            if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
                return None
            mask = metadata.get("qwen_attention_mask")
            if (
                isinstance(mask, torch.Tensor)
                and mask.ndim == 1
                and int(mask.numel()) == int(ids.numel())
            ):
                ids = ids[: int(mask.sum())]
            ids_list.append(ids.to(dtype=torch.long).contiguous())
        return ids_list

    @staticmethod
    def _apply_post_encoder_text_drop(
        cond_tokens: torch.Tensor,
        cond_meta: dict,
        packed_text_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard-gather Qwen rows selected by the post-encoder semantic mask.

        Ported verbatim from the REVE trainer. Without a keep mask this is
        the legacy identity path (caller still enforces the strict budget
        check); with one, two cross-checks raise on tokenizer or budget
        drift, and ``text_pos_meta`` is left-compacted so bbox row *i* stays
        aligned with kept token *i* after flattening.
        """
        full_mask = cond_meta["text_embedding_mask"].to(dtype=torch.bool)
        if cond_tokens.ndim not in (3, 4) or cond_tokens.shape[:2] != full_mask.shape:
            raise ValueError(
                "Qwen tokens and text mask differ before post-encoder drop: "
                f"tokens={tuple(cond_tokens.shape)} mask={tuple(full_mask.shape)}"
            )
        keep_mask = cond_meta.get("post_encoder_text_keep_mask")
        if keep_mask is None:
            realized = full_mask.sum(dim=1).to(dtype=torch.int32, device="cpu")
            return full_mask, realized
        if not isinstance(keep_mask, torch.Tensor):
            raise TypeError("post_encoder_text_keep_mask must be a tensor")
        keep_mask = keep_mask.to(device=full_mask.device, dtype=torch.bool)
        if (
            keep_mask.ndim != 2
            or keep_mask.shape[0] != full_mask.shape[0]
            or keep_mask.shape[1] > full_mask.shape[1]
        ):
            raise ValueError(
                "post-encoder keep mask is incompatible with Qwen mask: "
                f"keep={tuple(keep_mask.shape)} qwen={tuple(full_mask.shape)}"
            )
        if keep_mask.shape[1] < full_mask.shape[1]:
            padded_keep_mask = torch.zeros_like(full_mask)
            padded_keep_mask[:, : keep_mask.shape[1]] = keep_mask
            keep_mask = padded_keep_mask
        full_realized = full_mask.sum(dim=1).to(dtype=torch.int32, device="cpu")
        expected_full = torch.tensor(
            [int(value) for value in cond_meta.get("qwen_full_text_len", [])],
            dtype=torch.int32,
        )
        if expected_full.numel() and not torch.equal(full_realized, expected_full):
            raise ValueError(
                "online Qwen full lengths differ from MDS tokenization: "
                f"realized={full_realized.tolist()} "
                f"expected={expected_full.tolist()}"
            )
        effective_mask = full_mask & keep_mask
        realized = effective_mask.sum(dim=1).to(dtype=torch.int32, device="cpu")
        if not torch.equal(realized, packed_text_lens.cpu()):
            raise ValueError(
                "post-encoder text lengths differ from MDS pack budget: "
                f"realized={realized.tolist()} "
                f"budget={packed_text_lens.tolist()}"
            )
        text_pos_meta = cond_meta.get("text_pos_meta")
        if isinstance(text_pos_meta, torch.Tensor):
            if (
                text_pos_meta.ndim != 3
                or text_pos_meta.shape[0] != effective_mask.shape[0]
            ):
                raise ValueError(
                    "text_pos_meta must be [B,L,D] for post-encoder drop, got "
                    f"{tuple(text_pos_meta.shape)}"
                )
            if text_pos_meta.shape[1] < effective_mask.shape[1]:
                raise ValueError(
                    "text_pos_meta is shorter than the post-encoder mask: "
                    f"meta={tuple(text_pos_meta.shape)} "
                    f"mask={tuple(effective_mask.shape)}"
                )
            compact = text_pos_meta.new_zeros(text_pos_meta.shape)
            compact[..., 2] = 1.0
            compact[..., 3] = 1.0
            for row in range(int(effective_mask.shape[0])):
                selected = text_pos_meta[row, : effective_mask.shape[1]][
                    effective_mask[row].to(device=text_pos_meta.device)
                ]
                compact[row, : selected.shape[0]] = selected
            cond_meta["text_pos_meta"] = compact
        text_corr_mem_k = cond_meta.get("text_corr_mem_k")
        if isinstance(text_corr_mem_k, torch.Tensor):
            if (
                text_corr_mem_k.ndim != 3
                or text_corr_mem_k.shape[0] != effective_mask.shape[0]
                or text_corr_mem_k.shape[1] < effective_mask.shape[1]
            ):
                raise ValueError(
                    "text_corr_mem_k must cover the post-encoder mask, got "
                    f"membership={tuple(text_corr_mem_k.shape)} "
                    f"mask={tuple(effective_mask.shape)}"
                )
            compact_mem = text_corr_mem_k.new_zeros(text_corr_mem_k.shape)
            for row in range(int(effective_mask.shape[0])):
                selected = text_corr_mem_k[
                    row, : effective_mask.shape[1]
                ][effective_mask[row].to(device=text_corr_mem_k.device)]
                compact_mem[row, : selected.shape[0]] = selected
            cond_meta["text_corr_mem_k"] = compact_mem
        return effective_mask, realized

    @staticmethod
    def _flatten_text_tokens(
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if text_tokens.ndim not in (3, 4):
            raise ValueError(
                "text_tokens must be [B,L,D] or stacked [B,L,K,D], "
                f"got {tuple(text_tokens.shape)}"
            )
        if text_mask.shape != text_tokens.shape[:2]:
            raise ValueError(
                f"text_mask shape {tuple(text_mask.shape)} does not match "
                f"text_tokens {tuple(text_tokens.shape)}"
            )
        return text_tokens[
            text_mask.to(device=text_tokens.device, dtype=torch.bool)
        ]

    @torch.no_grad()
    def _encode_varlen_images(self, batch: Any) -> torch.Tensor:
        encode_packed = getattr(self.vae, "encode_packed", None)
        if callable(encode_packed):
            return encode_packed(batch.img_patches, batch.img_grids)
        return self.vae.encode(batch.img_patches)

    @torch.no_grad()
    def _compute_cond_bundle(
        self,
        batch: Any,
        device: torch.device,
        want_null: bool,
        x: torch.Tensor | None = None,
    ) -> tuple[Any, ...]:
        if x is None:
            x = self._encode_varlen_images(batch)

        cond_metadata: dict[str, Any] = {}
        text_lens = batch.text_lens
        text_tokens = batch.text_tokens
        null_flat = None
        multimodal_text_null = bool(
            want_null
            and getattr(
                self.diffusion_trainer, "cfg_dropout_mode", "null"
            )
            == "multimodal_text_null"
        )
        if text_tokens is None:
            metadata = self._stack_packed_metadata(batch.metadata) or {}
            cond_meta = dict(metadata)
            null_meta: dict[str, Any] | None = None
            null_captions: list[str] | None = None
            if multimodal_text_null:
                required = (
                    "cfg_null_caption",
                    "cfg_null_qwen_input_ids",
                    "cfg_null_qwen_attention_mask",
                    "cfg_null_text_pos_meta",
                    "cfg_null_qwen_text_len",
                )
                missing = [key for key in required if key not in metadata]
                if missing:
                    raise ValueError(
                        "multimodal text-null CFG metadata is incomplete: "
                        f"{missing}"
                    )
                null_captions = [str(value) for value in metadata["cfg_null_caption"]]
                null_meta = dict(metadata)
                null_meta["qwen_input_ids"] = metadata[
                    "cfg_null_qwen_input_ids"
                ]
                null_meta["qwen_attention_mask"] = metadata[
                    "cfg_null_qwen_attention_mask"
                ]
                null_meta["text_pos_meta"] = metadata[
                    "cfg_null_text_pos_meta"
                ]
                for key in (
                    "post_encoder_text_keep_mask",
                    "qwen_full_text_len",
                    "text_corr_mem_k",
                ):
                    null_meta.pop(key, None)
            if (
                getattr(self.conditioner, "use_token_packing", False)
                and "input_ids_list" not in cond_meta
            ):
                ragged = self._ragged_input_ids_from_pack(batch.metadata)
                if ragged is not None:
                    cond_meta["input_ids_list"] = ragged
            with fp8_autocast_ctx(enabled=self._te_cond_fp8_applied):
                cond_tokens = self.conditioner._impl_condition(
                    batch.captions,
                    cond_meta,
                ).to(device=device)
            text_mask, realized = self._apply_post_encoder_text_drop(
                cond_tokens, cond_meta, batch.text_lens
            )
            if "post_encoder_text_keep_mask" not in cond_meta and not torch.equal(
                realized, batch.text_lens.cpu()
            ):
                raise ValueError(
                    "online Qwen text lengths differ from MDS pack budget: "
                    f"realized={realized.tolist()} budget={batch.text_lens.tolist()}"
                )
            text_tokens = self._flatten_text_tokens(cond_tokens, text_mask)
            # Boolean indexing above owns its storage. The dense fixed-width
            # features are no longer needed, especially during the null pass.
            del cond_tokens
            cond_metadata = cond_meta
            text_lens = realized
            if multimodal_text_null:
                assert null_meta is not None and null_captions is not None
                with fp8_autocast_ctx(enabled=self._te_cond_fp8_applied):
                    null_tokens = self.conditioner._impl_condition(
                        null_captions,
                        null_meta,
                    ).to(device=device)
                null_mask = null_meta["text_embedding_mask"].to(
                    device=null_tokens.device, dtype=torch.bool
                )
                null_lens = null_mask.sum(dim=1).to(
                    dtype=torch.int32, device="cpu"
                )
                expected_null_lens = torch.tensor(
                    [int(value) for value in metadata["cfg_null_qwen_text_len"]],
                    dtype=torch.int32,
                )
                if not torch.equal(null_lens, expected_null_lens):
                    raise ValueError(
                        "V7-B online null Qwen lengths differ from packed budget: "
                        f"realized={null_lens.tolist()} "
                        f"expected={expected_null_lens.tolist()}"
                    )
                null_flat = {
                    "text_tokens": self._flatten_text_tokens(
                        null_tokens, null_mask
                    ),
                    "text_lens": null_lens,
                    "text_pos_meta": null_meta.get("text_pos_meta"),
                }
            elif want_null:
                get_null = getattr(
                    self.conditioner,
                    "_ensure_cached_uncondition",
                    None,
                )
                if callable(get_null):
                    # First call runs the encoder (cached after); keep it on
                    # the same numeric path as the conditional embeddings so
                    # CFG stays calibrated under TE FP8.
                    with fp8_autocast_ctx(enabled=self._te_cond_fp8_applied):
                        null_tokens, _null_mask, null_len_tensor = get_null()
                    null_len = int(null_len_tensor[0].item())
                    null_flat = null_tokens[0, :null_len].to(
                        device=device,
                        dtype=text_tokens.dtype,
                    )
                else:
                    with fp8_autocast_ctx(enabled=self._te_cond_fp8_applied):
                        _uncond_tokens, uncondition = self.conditioner(
                            batch.captions,
                            {},
                        )
                    null_len = int(uncondition.text_len[0].item())
                    null_flat = uncondition.text_tokens[0, :null_len].to(
                        device=device,
                        dtype=text_tokens.dtype,
                    )
        elif multimodal_text_null:
            raise ValueError(
                "multimodal text-null CFG requires online Qwen conditioning"
            )
        elif want_null:
            null_embedding = getattr(self.conditioner, "null_embedding", None)
            null_valid = getattr(self.conditioner, "null_valid_len", None)
            if null_embedding is not None and null_valid is not None:
                null_flat = null_embedding[0, : int(null_valid)].to(
                    device=device,
                    dtype=text_tokens.dtype,
                )
        return x, text_tokens, text_lens, null_flat, cond_metadata

    def _training_step_varlen(
        self,
        batch: Any,
        state: StepState,
    ) -> StepOutput:
        trainer = self.diffusion_trainer
        if not hasattr(trainer, "varlen_trainstep"):
            raise TypeError(
                "PackedVarlenBatch requires diffusion_trainer.varlen_trainstep"
            )
        strict = bool(getattr(self.denoiser, "strict_pack_target", False))
        network_pack_target = getattr(self.denoiser, "pack_target_T", None)
        if strict and int(network_pack_target) != int(batch.pack_target):
            raise ValueError(
                f"denoiser.pack_target_T={network_pack_target} != "
                f"dataset pack_target_T={batch.pack_target}"
            )

        device = batch.img_patches.device
        metadata = self._stack_packed_metadata(batch.metadata) or {}
        metadata["global_step"] = int(state.global_step)
        metadata["micro_step_in_accumulation"] = int(
            state.metadata.get("micro_step_in_accumulation", 0)
        )
        timing_on = device.type == "cuda" and is_enabled()
        if timing_on:
            begin_step(self)
            record_event(self, "start")
        want_null = float(getattr(trainer, "null_condition_p", 0.0)) > 0
        bundle = getattr(batch, "_async_cond_bundle", None)
        vae_encode_ms: float | None = None
        if bundle is not None:
            vae_event = getattr(batch, "_async_vae_event", None)
            if timing_on and vae_event is not None:
                torch.cuda.current_stream().wait_event(vae_event)
                record_event(self, "after_vae")
            event = getattr(batch, "_async_cond_event", None)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            vae_events = getattr(batch, "_async_vae_events", None)
            if (
                isinstance(vae_events, tuple)
                and len(vae_events) == 2
            ):
                try:
                    vae_events[1].synchronize()
                    vae_encode_ms = float(
                        vae_events[0].elapsed_time(vae_events[1])
                    )
                except Exception:
                    pass
            x, text_tokens, text_lens, null_flat, cond_metadata = bundle
            metadata.update(cond_metadata)
            # Keep tensors alive on the consumer stream, including nested
            # cond_metadata. The loader normally shares this stream, but a
            # caller may consume a batch in a different stream context.
            record_cuda_tensors(bundle, torch.cuda.current_stream())
            if timing_on:
                if vae_event is None:
                    record_event(self, "after_vae")
                record_event(self, "after_cond")
        else:
            vae_events = None
            if device.type == "cuda":
                vae_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                vae_events[0].record()
            with torch.no_grad():
                x = self._encode_varlen_images(batch)
            if vae_events is not None:
                vae_events[1].record()
            if timing_on:
                record_event(self, "after_vae")
            x, text_tokens, text_lens, null_flat, cond_metadata = (
                self._compute_cond_bundle(
                    batch,
                    device,
                    want_null,
                    x=x,
                )
            )
            metadata.update(cond_metadata)
            if vae_events is not None:
                try:
                    vae_events[1].synchronize()
                    vae_encode_ms = float(
                        vae_events[0].elapsed_time(vae_events[1])
                    )
                except Exception:
                    pass
            if timing_on:
                record_event(self, "after_cond")

        rgba_host = torch.tensor(
            [bool(item.get("is_rgba", False)) for item in batch.metadata],
            dtype=torch.bool,
        )
        is_rgba = (
            rgba_host.pin_memory().to(device, non_blocking=True)
            if device.type == "cuda"
            else rgba_host
        )
        loss = trainer.varlen_trainstep(
            self.denoiser,
            self.ema_denoiser,
            img_patches=x,
            text_tokens=text_tokens,
            text_lens=text_lens,
            img_grids=batch.img_grids,
            max_seqlen=batch.max_seqlen,
            pack_target=int(batch.pack_target),
            null_embedding=null_flat,
            metadata=metadata,
            is_rgba=is_rgba,
            img_seg_sample_id=getattr(batch, "img_seg_sample_id", None),
            img_seg_frame=getattr(batch, "img_seg_frame", None),
            img_seg_is_target=getattr(batch, "img_seg_is_target", None),
            image_corr_mem_q=getattr(batch, "image_corr_mem_q", None),
            image_corr_mem_k=getattr(batch, "image_corr_mem_k", None),
            image_token_roles=getattr(batch, "image_token_roles", None),
            corr_code_num_groups=getattr(
                batch, "corr_code_num_groups", None
            ),
        )
        if timing_on:
            record_event(self, "after_diff")
        t_bucket_metrics = loss.pop("_t_bucket_metrics", {})
        pack_logs: dict[str, float] = {
            "n_samples": float(batch.n_samples),
            "fill_rate": float(batch.total_tokens)
            / max(1.0, float(batch.pack_target)),
            "text_tokens": float(sum(int(value) for value in text_lens.tolist())),
            "img_tokens": float(batch.img_patches.shape[0]),
        }
        if "post_encoder_text_keep_mask" in metadata:
            full_lens = metadata.get("qwen_full_text_len") or []
            full_total = float(sum(int(value) for value in full_lens))
            if full_total > 0:
                kept_total = float(sum(int(value) for value in text_lens.tolist()))
                pack_logs["qwen_full_text_tokens"] = full_total
                pack_logs["post_encoder_text_keep_ratio"] = kept_total / full_total
        if batch.packer_stats:
            stats = {
                str(key): float(value) for key, value in batch.packer_stats.items()
            }
            drops = math.fsum(
                value for key, value in stats.items() if key.endswith("_dropped")
            )
            pack_logs["drops_total"] = drops
            if debug_metrics_enabled():
                pack_logs["pad_tokens"] = float(batch.pad_tokens)
                pack_logs.update(stats)
            else:
                always_log = (
                    "exposed_ms",
                    "buf_samples",
                    "corr_code_groups_total_current",
                    "corr_code_groups_max_current",
                    "corr_code_candidate_edges_total_current",
                    "corr_code_pruned_edges_total_current",
                    "corr_code_coded_samples_current",
                    "corr_code_null_samples_current",
                    "corr_code_graph_disabled_samples_current",
                    "corr_code_graph_disabled_null_current",
                    "corr_code_graph_disabled_unroutable_current",
                    "corr_code_graph_disabled_text_pruned_current",
                    "corr_code_graph_pruned_samples_current",
                    "corr_code_graph_pruned_invalid_reference_current",
                    "corr_code_graph_pruned_insufficient_retained_bbox_area_current",
                    "corr_code_graph_pruned_missing_instruction_span_current",
                    "corr_code_graph_pruned_incomplete_carrier_current",
                    "corr_code_image_tokens_current",
                    "corr_code_image_memberships_current",
                    "corr_code_image_membership_weight_sum_current",
                    "corr_code_memberships_truncated_current",
                    "corr_code_text_tokens_current",
                )
                corr_observability = tuple(
                    f"corr_code_{role}_{metric}_current"
                    for role in ("image_q", "image_k", "text_k")
                    for metric in (
                        "degree_max",
                        "degree_gt8",
                        "degree_gt16",
                        "degree_gt32",
                        "degree_gt64",
                        "energy_max",
                        "energy_p95",
                    )
                )
                for key in (*always_log, *corr_observability):
                    if key in stats:
                        pack_logs[key] = stats[key]
        # Bare trainer keys land in W&B's root "Charts" section; group them
        # under train/. The runner already emits train/loss from the registry,
        # so the task's duplicate "loss" scalar is dropped.
        logs = {
            (key if "/" in key else f"train/{key}"): value
            for key, value in loss.items()
            if key != "loss"
        }
        if vae_encode_ms is not None:
            logs["perf/vae_encode_ms"] = vae_encode_ms
        return StepOutput(
            loss=loss["loss"],
            logs=logs,
            aux_logs={
                "t_bucket": dict(t_bucket_metrics),
                "data": pack_logs,
            },
            metadata={
                "throughput": {
                    "samples": float(batch.n_samples),
                    "total_tokens": float(batch.total_tokens),
                }
            },
        )

    def state_dict(
        self,
        *args: Any,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Exclude frozen conditioner/VAE weights from native checkpoints."""

        del args
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=f"{prefix}denoiser.",
            keep_vars=keep_vars,
        )
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=f"{prefix}ema_denoiser.",
            keep_vars=keep_vars,
        )
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=f"{prefix}diffusion_trainer.",
            keep_vars=keep_vars,
        )
        return destination

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Strictly restore checkpoint-owned state while keeping frozen trunks.

        Native checkpoints deliberately omit the pretrained VAE and conditioner,
        which are materialized from their configured source before checkpoint
        loading.  Treat only those two prefixes as expected omissions; every
        trainable/EMA/trainer key and every unexpected key remains strict.
        """

        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        if not strict:
            return incompatible

        allowed_missing_prefixes = ("vae.", "conditioner.")
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"Missing key(s) in state_dict: {missing!r}.")
            if unexpected:
                details.append(
                    f"Unexpected key(s) in state_dict: {unexpected!r}."
                )
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{type(self).__name__}:\n\t" + "\n\t".join(details)
            )
        return type(incompatible)([], [])


__all__ = [
    "OptimizerFactory",
    "PixelCraftNativeTask",
    "SchedulerFactory",
]
