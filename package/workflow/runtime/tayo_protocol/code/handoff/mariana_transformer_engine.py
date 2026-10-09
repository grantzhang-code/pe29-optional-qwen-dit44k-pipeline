"""TransformerEngine FP8 integration.

Phase 3 hoists the actual ``nn.Linear`` → ``te.Linear`` replacement
into mariana so:

  * any task can opt into TE FP8 without a task-side helper,
  * trainable replacement (main network) is supported alongside the
    frozen path that PixelCraft historically used for ``ema_denoiser``,
  * the optional FP8 ``amax`` sync group can be supplied (FSDP2 shard
    submesh, future megatron-core TP / DP group, …) so amax stays
    consistent across the shards that share parameters.

Env vars (PIXELCRAFT_* names kept verbatim for backward compat;
non-PixelCraft tasks can read the underlying ``is_te_fp8_enabled``
value directly):

  PIXELCRAFT_TE_FP8=1            globally enable TE FP8
  PIXELCRAFT_TE_REPLACE_MAIN=1   replace main network too (default
                                  off — safer baseline)
  PIXELCRAFT_TE_INCLUDE=regex    comma-separated module-name regex allowlist
  PIXELCRAFT_TE_EXCLUDE=regex    comma-separated module-name regex denylist
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, ContextManager, Iterator, Mapping, Optional

import torch
import torch.nn as nn

from mariana.integrations.availability import (
    IntegrationAvailability,
    transformer_engine_availability,
)


_TE_FP8_ENV = "PIXELCRAFT_TE_FP8"
_TE_REPLACE_MAIN_ENV = "PIXELCRAFT_TE_REPLACE_MAIN"
_TE_INCLUDE_ENV = "PIXELCRAFT_TE_INCLUDE"
_TE_EXCLUDE_ENV = "PIXELCRAFT_TE_EXCLUDE"
_TE_REPLACE_EMA_ENV = "PIXELCRAFT_TE_REPLACE_EMA"
_TE_REPLACE_COND_ENV = "PIXELCRAFT_TE_REPLACE_COND"
_TE_COND_INCLUDE_ENV = "PIXELCRAFT_TE_COND_INCLUDE"
_TE_COND_EXCLUDE_ENV = "PIXELCRAFT_TE_COND_EXCLUDE"
_TE_COND_FP8_WEIGHTS_ENV = "PIXELCRAFT_TE_COND_FP8_WEIGHTS"


def _split_env_patterns(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _coerce_patterns(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _split_env_patterns(value)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise TypeError(f"{field_name} must be a string or sequence of strings")


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return False
    raise TypeError(f"{field_name} must be a boolean-like value")


def is_te_fp8_enabled(
    config: "TEIntegrationConfig | Mapping[str, Any] | None" = None,
) -> bool:
    """``PIXELCRAFT_TE_FP8=1`` flag."""
    return resolve_te_integration_config(config).enabled


def should_replace_main_network(
    config: "TEIntegrationConfig | Mapping[str, Any] | None" = None,
) -> bool:
    """``PIXELCRAFT_TE_REPLACE_MAIN=1`` — opt-in for trainable main net.

    Off by default in Phase 3 because the FFA + TE interaction is not
    yet validated in production. Enable per-run once an overlay run
    confirms loss equivalence.
    """
    return resolve_te_integration_config(config).replace_main_network


def should_replace_ema(
    config: "TEIntegrationConfig | Mapping[str, Any] | None" = None,
) -> bool:
    """EMA replacement gate. Defaults ON to preserve the historical
    behaviour (``enabled: true`` alone converts the frozen EMA copy);
    set ``replace_ema: false`` to run a conditioner-only TE ablation
    without touching EMA sampling numerics."""
    return resolve_te_integration_config(config).replace_ema


def should_replace_conditioner(
    config: "TEIntegrationConfig | Mapping[str, Any] | None" = None,
) -> bool:
    """Opt-in FP8 replacement of the frozen text-conditioner tower."""
    return resolve_te_integration_config(config).replace_conditioner


class TEIntegrationUnavailableError(ImportError):
    """Raised when TE FP8 is requested but TransformerEngine cannot import."""

    def __init__(
        self,
        message: str,
        *,
        availability: IntegrationAvailability | None = None,
    ) -> None:
        super().__init__(message)
        self.availability = availability


def _get_te():
    try:
        import transformer_engine.pytorch as te
    except ImportError as exc:
        availability = transformer_engine_availability()
        errors = "; ".join(availability.errors) or str(exc)
        raise TEIntegrationUnavailableError(
            "TransformerEngine is required for TE FP8 replacement but could not "
            f"be imported ({errors})",
            availability=availability,
        ) from exc
    return te


@dataclass(frozen=True)
class TEReplacePolicy:
    """Name-based TE Linear replacement policy.

    ``include`` and ``exclude`` are regex patterns matched against the
    fully-qualified module name seen during recursion (for example
    ``"blocks.0.attn.qkv"`` or ``"final_layer.linear"``). Empty
    ``include`` means all ``nn.Linear`` modules are eligible. ``exclude``
    always wins.
    """

    include: tuple[str, ...] = field(default_factory=tuple)
    exclude: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> "TEReplacePolicy":
        return cls(
            include=_split_env_patterns(os.environ.get(_TE_INCLUDE_ENV)),
            exclude=_split_env_patterns(os.environ.get(_TE_EXCLUDE_ENV)),
        )

    def should_replace(self, module_name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if self.include and not any(
            re.search(pattern, module_name) for pattern in self.include
        ):
            return False
        if self.exclude and any(
            re.search(pattern, module_name) for pattern in self.exclude
        ):
            return False
        return True


@dataclass(frozen=True)
class TEIntegrationConfig:
    """Structured TransformerEngine integration config.

    This is the config-file counterpart to the legacy ``PIXELCRAFT_TE_*`` env
    flags. It intentionally stays small: enablement, trainable-main opt-in, and
    module-name include/exclude filters. Constructor overrides and dtype/shape
    predicates can be added once real TE ablations need them.
    """

    enabled: bool = False
    replace_main_network: bool = False
    include: tuple[str, ...] = field(default_factory=tuple)
    exclude: tuple[str, ...] = field(default_factory=tuple)
    # EMA replacement stays the historical default; conditioner replacement
    # is a separate opt-in with its own name-pattern policy because the two
    # models share no module names.
    replace_ema: bool = True
    replace_conditioner: bool = False
    conditioner_include: tuple[str, ...] = field(default_factory=tuple)
    conditioner_exclude: tuple[str, ...] = field(default_factory=tuple)
    # Store the conditioner's replaced weights directly in FP8
    # (``te.fp8_model_init``): halves the frozen tower's resident memory and
    # skips the per-forward weight quantize. Only valid for frozen modules.
    conditioner_fp8_weights: bool = True

    def __post_init__(self) -> None:
        for field_name, patterns in (
            ("include", self.include),
            ("exclude", self.exclude),
            ("conditioner_include", self.conditioner_include),
            ("conditioner_exclude", self.conditioner_exclude),
        ):
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"invalid TE {field_name} regex {pattern!r}: {exc}"
                    ) from exc

    @classmethod
    def from_env(cls) -> "TEIntegrationConfig":
        return cls(
            enabled=os.environ.get(_TE_FP8_ENV, "0") == "1",
            replace_main_network=os.environ.get(_TE_REPLACE_MAIN_ENV, "0") == "1",
            include=_split_env_patterns(os.environ.get(_TE_INCLUDE_ENV)),
            exclude=_split_env_patterns(os.environ.get(_TE_EXCLUDE_ENV)),
            replace_ema=os.environ.get(_TE_REPLACE_EMA_ENV, "1") == "1",
            replace_conditioner=os.environ.get(_TE_REPLACE_COND_ENV, "0") == "1",
            conditioner_include=_split_env_patterns(
                os.environ.get(_TE_COND_INCLUDE_ENV)
            ),
            conditioner_exclude=_split_env_patterns(
                os.environ.get(_TE_COND_EXCLUDE_ENV)
            ),
            conditioner_fp8_weights=os.environ.get(
                _TE_COND_FP8_WEIGHTS_ENV, "1"
            )
            == "1",
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TEIntegrationConfig":
        policy = raw.get("policy", {})
        if policy is None:
            policy = {}
        if not isinstance(policy, Mapping):
            raise TypeError("TE policy must be a mapping when provided")
        include = raw.get("include", policy.get("include"))
        exclude = raw.get("exclude", policy.get("exclude"))
        conditioner = raw.get("conditioner", {})
        if conditioner is None:
            conditioner = {}
        if not isinstance(conditioner, Mapping):
            raise TypeError("TE conditioner config must be a mapping when provided")
        return cls(
            enabled=_coerce_bool(raw.get("enabled", False), field_name="enabled"),
            replace_main_network=_coerce_bool(
                raw.get(
                    "replace_main_network",
                    raw.get("replace_main", False),
                ),
                field_name="replace_main_network",
            ),
            include=_coerce_patterns(include, field_name="include"),
            exclude=_coerce_patterns(exclude, field_name="exclude"),
            replace_ema=_coerce_bool(
                raw.get("replace_ema", True), field_name="replace_ema"
            ),
            replace_conditioner=_coerce_bool(
                conditioner.get("replace", False),
                field_name="conditioner.replace",
            ),
            conditioner_include=_coerce_patterns(
                conditioner.get("include"), field_name="conditioner_include"
            ),
            conditioner_exclude=_coerce_patterns(
                conditioner.get("exclude"), field_name="conditioner_exclude"
            ),
            conditioner_fp8_weights=_coerce_bool(
                conditioner.get("fp8_weights", True),
                field_name="conditioner.fp8_weights",
            ),
        )

    @property
    def policy(self) -> TEReplacePolicy:
        return TEReplacePolicy(include=self.include, exclude=self.exclude)

    @property
    def conditioner_policy(self) -> TEReplacePolicy:
        return TEReplacePolicy(
            include=self.conditioner_include,
            exclude=self.conditioner_exclude,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "replace_main_network": self.replace_main_network,
            "replace_ema": self.replace_ema,
            "policy": {
                "include": list(self.include),
                "exclude": list(self.exclude),
            },
            "conditioner": {
                "replace": self.replace_conditioner,
                "fp8_weights": self.conditioner_fp8_weights,
                "include": list(self.conditioner_include),
                "exclude": list(self.conditioner_exclude),
            },
        }


def resolve_te_integration_config(
    config: TEIntegrationConfig | Mapping[str, Any] | None = None,
) -> TEIntegrationConfig:
    if config is None:
        return TEIntegrationConfig.from_env()
    if isinstance(config, TEIntegrationConfig):
        return config
    if isinstance(config, Mapping):
        return TEIntegrationConfig.from_mapping(config)
    raise TypeError(
        "TE config must be None, TEIntegrationConfig, or a mapping; "
        f"got {type(config).__name__}"
    )


@dataclass(frozen=True)
class TEApplyReport:
    """Structured debug report for one TE replacement pass."""

    model_type: str
    replaced_modules: tuple[str, ...]
    frozen: bool
    policy: TEReplacePolicy
    amax_sync_group_attached: bool

    @property
    def replaced_count(self) -> int:
        return len(self.replaced_modules)

    def to_json(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "replaced_count": self.replaced_count,
            "replaced_modules": list(self.replaced_modules),
            "frozen": self.frozen,
            "policy": {
                "include": list(self.policy.include),
                "exclude": list(self.policy.exclude),
            },
            "amax_sync_group_attached": self.amax_sync_group_attached,
        }


@dataclass(frozen=True)
class TEReplacementPlanReport:
    """Dry-run report for TE Linear replacement policy decisions."""

    model_type: str
    enabled: bool
    replace_main_network: bool
    frozen: bool
    policy: TEReplacePolicy
    eligible_modules: tuple[str, ...]
    skipped_modules: tuple[str, ...]
    existing_te_modules: tuple[str, ...]
    module_details: dict[str, dict[str, Any]]
    availability: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_modules)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_modules)

    @property
    def existing_te_count(self) -> int:
        return len(self.existing_te_modules)

    @property
    def replacement_active(self) -> bool:
        return self.enabled and self.eligible_count > 0

    def to_json(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "enabled": self.enabled,
            "replace_main_network": self.replace_main_network,
            "frozen": self.frozen,
            "policy": {
                "include": list(self.policy.include),
                "exclude": list(self.policy.exclude),
            },
            "replacement_active": self.replacement_active,
            "eligible_count": self.eligible_count,
            "skipped_count": self.skipped_count,
            "existing_te_count": self.existing_te_count,
            "eligible_modules": list(self.eligible_modules),
            "skipped_modules": list(self.skipped_modules),
            "existing_te_modules": list(self.existing_te_modules),
            "module_details": self.module_details,
            "availability": self.availability,
        }


class TEReplacementValidationError(ValueError):
    """Raised when actual TE replacement does not match the planned policy."""


@dataclass(frozen=True)
class TEReplacementValidationReport:
    """Validation report comparing a TE dry-run plan with actual replacement."""

    model_type: str
    frozen: bool
    ok: bool
    require_replacement: bool
    expected_modules: tuple[str, ...]
    replaced_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]
    unexpected_modules: tuple[str, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def raise_for_error(self) -> None:
        if self.ok:
            return
        detail = "\n".join(self.issues)
        raise TEReplacementValidationError(
            f"TE replacement validation failed for {self.model_type}:\n{detail}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "frozen": self.frozen,
            "ok": self.ok,
            "require_replacement": self.require_replacement,
            "issue_count": self.issue_count,
            "expected_modules": list(self.expected_modules),
            "replaced_modules": list(self.replaced_modules),
            "missing_modules": list(self.missing_modules),
            "unexpected_modules": list(self.unexpected_modules),
            "issues": list(self.issues),
        }


def validate_te_replacement(
    plan: TEReplacementPlanReport,
    report: TEApplyReport,
    *,
    require_replacement: bool = True,
    raise_on_error: bool = False,
) -> TEReplacementValidationReport:
    """Validate that actual TE replacement matches a prior dry-run plan."""

    missing_modules = tuple(
        name for name in plan.eligible_modules if name not in report.replaced_modules
    )
    unexpected_modules = tuple(
        name for name in report.replaced_modules if name not in plan.eligible_modules
    )
    issues: list[str] = []
    if plan.model_type != report.model_type:
        issues.append(
            f"model type changed between plan ({plan.model_type}) "
            f"and apply ({report.model_type})"
        )
    if plan.frozen != report.frozen:
        issues.append(
            f"frozen flag changed between plan ({plan.frozen}) "
            f"and apply ({report.frozen})"
        )
    if (
        plan.enabled
        and require_replacement
        and not plan.eligible_modules
        and not plan.existing_te_modules
    ):
        issues.append(
            "TE FP8 is enabled but the replacement policy matched no "
            "nn.Linear modules"
        )
    if missing_modules:
        issues.append(
            "TE apply missed planned module(s): "
            + ", ".join(missing_modules)
        )
    if unexpected_modules:
        issues.append(
            "TE apply replaced unplanned module(s): "
            + ", ".join(unexpected_modules)
        )

    validation = TEReplacementValidationReport(
        model_type=plan.model_type,
        frozen=plan.frozen,
        ok=not issues,
        require_replacement=require_replacement,
        expected_modules=plan.eligible_modules,
        replaced_modules=report.replaced_modules,
        missing_modules=missing_modules,
        unexpected_modules=unexpected_modules,
        issues=tuple(issues),
    )
    if raise_on_error:
        validation.raise_for_error()
    return validation


def plan_te_fp8_replacement(
    model: nn.Module,
    *,
    frozen: bool = True,
    policy: Optional[TEReplacePolicy] = None,
    config: TEIntegrationConfig | Mapping[str, Any] | None = None,
    include_availability: bool = False,
) -> TEReplacementPlanReport:
    """Dry-run TE replacement without importing TE or mutating ``model``."""

    resolved_config = resolve_te_integration_config(config)
    if policy is None:
        policy = resolved_config.policy

    eligible_modules: list[str] = []
    skipped_modules: list[str] = []
    existing_te_modules: list[str] = []
    module_details: dict[str, dict[str, Any]] = {}

    for name, module in model.named_modules():
        if not name:
            continue
        if _looks_like_te_linear(module):
            existing_te_modules.append(name)
            module_details[name] = _linear_module_details(module)
            continue
        if not isinstance(module, nn.Linear):
            continue
        module_details[name] = _linear_module_details(module)
        if policy.should_replace(name, module):
            eligible_modules.append(name)
        else:
            skipped_modules.append(name)

    availability = (
        transformer_engine_availability().to_json() if include_availability else {}
    )
    return TEReplacementPlanReport(
        model_type=type(model).__name__,
        enabled=resolved_config.enabled,
        replace_main_network=resolved_config.replace_main_network,
        frozen=frozen,
        policy=policy,
        eligible_modules=tuple(eligible_modules),
        skipped_modules=tuple(skipped_modules),
        existing_te_modules=tuple(existing_te_modules),
        module_details=module_details,
        availability=availability,
    )


def _looks_like_te_linear(module: nn.Module) -> bool:
    module_name = type(module).__module__
    class_name = type(module).__name__.lower()
    if "transformer_engine" in module_name and class_name.endswith("linear"):
        return True
    return hasattr(module, "fp8_meta") and hasattr(module, "_fp8_amax_reduce_group")


def _linear_module_details(module: nn.Module) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    details: dict[str, Any] = {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "in_features": getattr(module, "in_features", None),
        "out_features": getattr(module, "out_features", None),
        "bias": bias is not None,
    }
    if weight is not None:
        details.update(
            {
                "weight_shape": list(weight.shape),
                "weight_dtype": str(weight.dtype),
                "weight_device": str(weight.device),
                "weight_requires_grad": bool(weight.requires_grad),
            }
        )
    if bias is not None:
        details.update(
            {
                "bias_shape": list(bias.shape),
                "bias_dtype": str(bias.dtype),
                "bias_device": str(bias.device),
                "bias_requires_grad": bool(bias.requires_grad),
            }
        )
    return details


def replace_linear_with_te(
    module: nn.Module,
    *,
    frozen: bool = True,
    name_prefix: str = "",
    policy: Optional[TEReplacePolicy] = None,
    fp8_weight_storage: bool = False,
) -> int:
    """Recursively swap ``nn.Linear`` with ``te.Linear`` in ``module``.

    Weight + bias are copied through; the new TE Linear inherits the
    original ``params_dtype``. ``frozen=True`` (default) zeroes
    ``requires_grad`` on the new params — used for the EMA copy and
    other frozen modules where the FP8 path is purely a forward
    speedup. ``frozen=False`` leaves grads enabled so the main network
    can train under TE FP8.

    ``fp8_weight_storage=True`` constructs the TE Linears under
    ``te.fp8_model_init`` so the replaced weights are STORED as FP8
    (half the resident memory, no per-forward weight quantize). Frozen
    modules only: a trainable module needs its high-precision master
    weights.

    Returns the number of layers replaced.
    """
    if fp8_weight_storage and not frozen:
        raise ValueError(
            "fp8_weight_storage=True requires frozen=True: a trainable "
            "module must keep high-precision master weights"
        )
    if policy is None:
        policy = TEReplacePolicy()
    replaced: list[str] = []
    if fp8_weight_storage:
        te = _get_te()
        from transformer_engine.common.recipe import Float8CurrentScaling
        with te.fp8_model_init(enabled=True, recipe=Float8CurrentScaling()):
            _replace_linear_with_te(
                module,
                frozen=frozen,
                name_prefix=name_prefix,
                policy=policy,
                replaced_modules=replaced,
            )
    else:
        _replace_linear_with_te(
            module,
            frozen=frozen,
            name_prefix=name_prefix,
            policy=policy,
            replaced_modules=replaced,
        )
    return len(replaced)


def _replace_linear_with_te(
    module: nn.Module,
    *,
    frozen: bool,
    name_prefix: str,
    policy: TEReplacePolicy,
    replaced_modules: list[str],
) -> None:
    te = _get_te()
    # A frozen module never runs backward, so its FP8 weight only needs the
    # rowwise copy. TE fixes the weight quantizer's usage at construction via
    # ``torch.is_grad_enabled()`` (module/base.py ``reset_parameters``:
    # ``set_usage(rowwise=True, columnwise=torch.is_grad_enabled())``), so
    # build frozen replacements under ``no_grad``. Otherwise
    # ``fp8_model_init`` also allocates the columnwise copy and MXFP8 storage
    # lands at bf16 parity (~2 bytes/element) instead of half. Trainable
    # modules keep grad enabled: dgrad needs the columnwise weight.
    build_ctx = torch.no_grad if frozen else nullcontext
    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(child, te.Linear):
            continue
        if policy.should_replace(full_name, child):
            with build_ctx():
                te_linear = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                )
            with torch.no_grad():
                te_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    te_linear.bias.copy_(child.bias)
            _copy_parameter_metadata(child.weight, te_linear.weight)
            if child.bias is not None and te_linear.bias is not None:
                _copy_parameter_metadata(child.bias, te_linear.bias)
            if frozen:
                te_linear.requires_grad_(False)
            setattr(module, name, te_linear)
            replaced_modules.append(full_name)
        else:
            _replace_linear_with_te(
                child,
                frozen=frozen,
                name_prefix=full_name,
                policy=policy,
                replaced_modules=replaced_modules,
            )


def _copy_parameter_metadata(src: nn.Parameter, dst: nn.Parameter) -> None:
    """Preserve optimizer-side parameter metadata across replacement.

    μP attaches ``param.infshape`` to parameters, and PixelCraft's Muon
    optimizers read it from param groups at construction / step time.
    Replacing a Linear after base-shape registration must therefore copy
    this attribute onto the new TE parameter; otherwise the FP8 path can
    silently lose μP scaling information. Keep this deliberately narrow:
    arbitrary Python attrs may hold module-specific state that should not
    be cloned to a different parameter object.
    """
    if hasattr(src, "infshape"):
        dst.infshape = src.infshape


def apply_te_fp8(
    model: nn.Module,
    *,
    frozen: bool = True,
    rank: int = 0,
    amax_sync_group: Optional[Any] = None,
    policy: Optional[TEReplacePolicy] = None,
    config: TEIntegrationConfig | Mapping[str, Any] | None = None,
    return_report: bool = False,
    fp8_weight_storage: bool = False,
) -> int | TEApplyReport:
    """Replace Linear layers and (optionally) wire FP8 amax sync.

    Parameters:
      model: target module (denoiser, ema_denoiser, ...).
      frozen: forwarded to ``replace_linear_with_te``; ``True`` for
        EMA-style frozen copies, ``False`` for trainable main networks.
      rank: caller's global rank. ``rank == 0`` is the only one that
        prints the per-model summary line, to avoid log spam under
        multi-GPU.
      amax_sync_group: optional ``torch.distributed`` ProcessGroup (or
        DeviceMesh submesh) across which FP8 ``amax`` history should be
        all-reduced each step. PixelCraft passes
        ``FSDP2Strategy.shard_submesh`` so shards of the same parameter
        agree on ``amax`` history. ``None`` defers to TE's default
        (per-process amax, fine for single-rank or DP-only).
      policy: optional ``TEReplacePolicy``. When omitted, the policy is read
        from ``config`` or from ``PIXELCRAFT_TE_INCLUDE`` /
        ``PIXELCRAFT_TE_EXCLUDE`` so production jobs can do targeted ablations
        without changing code.
      config: optional structured TE config mapping/dataclass. Explicit
        ``policy`` wins over ``config.policy`` for backward compatibility.
      return_report: when True, return ``TEApplyReport`` instead of the
        legacy integer replacement count.

    Returns the number of swapped layers unless ``return_report=True``.
    """
    resolved_config = resolve_te_integration_config(config)
    if policy is None:
        policy = resolved_config.policy
    replaced_modules: list[str] = []
    if fp8_weight_storage and not frozen:
        raise ValueError(
            "fp8_weight_storage=True requires frozen=True: a trainable "
            "module must keep high-precision master weights"
        )
    if fp8_weight_storage:
        te = _get_te()
        from transformer_engine.common.recipe import Float8CurrentScaling
        with te.fp8_model_init(enabled=True, recipe=Float8CurrentScaling()):
            _replace_linear_with_te(
                model,
                frozen=frozen,
                name_prefix="",
                policy=policy,
                replaced_modules=replaced_modules,
            )
    else:
        _replace_linear_with_te(
            model,
            frozen=frozen,
            name_prefix="",
            policy=policy,
            replaced_modules=replaced_modules,
        )
    n = len(replaced_modules)
    if amax_sync_group is not None and n > 0:
        _attach_amax_sync_group(model, amax_sync_group)
    report = TEApplyReport(
        model_type=type(model).__name__,
        replaced_modules=tuple(replaced_modules),
        frozen=frozen,
        policy=policy,
        amax_sync_group_attached=amax_sync_group is not None and n > 0,
    )
    if rank == 0 and n > 0:
        kind = "frozen" if frozen else "trainable"
        sync = f", amax_sync_group attached" if amax_sync_group is not None else ""
        print(
            f"[te_fp8] Replaced {n} nn.Linear → te.Linear in "
            f"{type(model).__name__} ({kind}){sync}",
            flush=True,
        )
    if return_report:
        return report
    return n


def _attach_amax_sync_group(model: nn.Module, group: Any) -> None:
    """Tag every TE module under ``model`` with the amax sync group.

    TE reads ``module.fp8_meta['fp8_group']`` (recent versions) /
    ``module._fp8_amax_reduce_group`` (older) on each step's amax
    reduction. Set both for forward-compat.
    """
    te = _get_te()
    pg = group
    if hasattr(group, "get_group"):
        # DeviceMesh submesh — extract underlying ProcessGroup.
        try:
            pg = group.get_group()
        except Exception:
            pg = group

    for sub in model.modules():
        if not isinstance(sub, te.Linear):
            continue
        if hasattr(sub, "fp8_meta") and isinstance(sub.fp8_meta, dict):
            sub.fp8_meta["fp8_group"] = pg
        try:
            sub._fp8_amax_reduce_group = pg
        except AttributeError:
            pass


@contextmanager
def fp8_autocast_ctx(enabled: Optional[bool] = None) -> Iterator[ContextManager]:
    """Context manager wrapping ``te.fp8_autocast`` when enabled.

    ``enabled=None`` resolves to ``is_te_fp8_enabled()`` so the caller
    can blanket-wrap forward / sampler paths without thinking about
    whether the env var is set.
    """
    if enabled is None:
        enabled = is_te_fp8_enabled()
    if not enabled:
        yield None
        return
    te = _get_te()
    from transformer_engine.common.recipe import Float8CurrentScaling
    with te.fp8_autocast(enabled=True, fp8_recipe=Float8CurrentScaling()) as ctx:
        yield ctx


__all__ = [
    "IntegrationAvailability",
    "TEApplyReport",
    "TEIntegrationConfig",
    "TEIntegrationUnavailableError",
    "TEReplacePolicy",
    "TEReplacementPlanReport",
    "TEReplacementValidationError",
    "TEReplacementValidationReport",
    "apply_te_fp8",
    "fp8_autocast_ctx",
    "is_te_fp8_enabled",
    "plan_te_fp8_replacement",
    "replace_linear_with_te",
    "resolve_te_integration_config",
    "should_replace_conditioner",
    "should_replace_ema",
    "should_replace_main_network",
    "transformer_engine_availability",
    "validate_te_replacement",
]
