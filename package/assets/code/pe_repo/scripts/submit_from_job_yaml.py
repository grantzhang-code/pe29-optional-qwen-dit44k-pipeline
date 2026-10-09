#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyyaml",
# ]
# ///
"""Submit an experiment through a workers-only raw UTP JobSet template.

The YAML is the experiment source of truth. This script resolves its environment
variables, validates the referenced raw JobSet, and delegates to the UTP executable
installed on PATH (or the checked-in core-cn skill fallback).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from string import Template
from typing import Any

import yaml


# Local UTP script shared by Core CN workflows on this devbox. Tests replace
# this path, and an installed `utp` on PATH still takes precedence.
UTP_SKILL_FALLBACK = Path("/mnt/ephemeral/pcb-v2-9c4d-17000/legacy_assets/utp")
UTP_SKILL_DEPENDENCIES = ("arrow", "click", "kubernetes", "pyyaml", "questionary", "rich")


OPTIONAL_ENV_VARS = (
    "DLMS_RUN_SCRIPT",
    "CUSTOM_REGISTER_PATH",
    "MODEL_S3_ARTIFACT",
    "MODEL_S3_ARTIFACT_PROTOCOL",
    "MODEL_S3_ARTIFACT_MANIFEST_SHA256",
    "MODEL_LOCAL_DIR",
    "DATASET_S3_URI",
    "DATASET_S3_SHA256",
    "DATASET_LOCAL_PATH",
    "TRAIN_ITERS",
    "NUM_TRAIN_EPOCHS",
    "RESUME_CKPT",
    "MCORE_S3_ROOT",
    "MCORE_S3_RESUME_URI",
    "MCORE_S3_ATTEMPT_ID",
    "MEGATRON_EXTRA",
    "CRAFT_MIXED4_CONFIG",
    "CRAFT_MIXED4_PRIMARY_PATHS",
    "CRAFT_MIXED4_REGION_PATHS",
    "CRAFT_HIERARCHY_DATASET_NAME",
    "CRAFT_HIERARCHY_VAL_DATASET_NAME",
    "CRAFT_HIERARCHY_VAL_PATH_COUNT",
    "CRAFT_HIERARCHY_VAL_MAX_SAMPLES",
    "CRAFT_HIERARCHY_SAMPLE_SHUFFLE_BUFFER",
    "CRAFT_HIERARCHY_DATA_SEED",
    "CRAFT_HIERARCHY_MAX_SAMPLES",
    "CRAFT_PACKING_REQUIRED_CHANNELS",
    "TEXT_TO_JSON_MANIFEST",
    "TEXT_TO_JSON_PATHS",
    "TEXT_TO_JSON_CONFIG",
    "TEXT_TO_JSON_RANK_SHARD",
    "TEXT_TO_JSON_REPEAT",
    "TEXT_TO_JSON_SHUFFLE_PATHS",
    "TEXT_TO_JSON_SAMPLE_SHUFFLE_BUFFER",
    "TEXT_TO_JSON_SEED",
    "TEXT_TO_JSON_MAX_SAMPLES",
    "RLHF_TYPE",
    "RESUME_FROM",
    "RESUME_ONLY_MODEL",
    "AUTO_RESUME_FROM_S3",
    "RESUME_REQUIRE_FULL_STATE",
    "MAX_RETRIES",
    "VLLM_ATTENTION_BACKEND",
    "TEST_IMAGES_DIR",
    "TEST_IMAGES_PROMPT",
    "OUTPUT_S3",
    "WANDB_RUN_ID",
    "WANDB_RUN_NAME",
    "WANDB_RESUME",
    "LAYOUT_JSONL_PATHS",
    "LAYOUT_IMAGES_ROOTS",
    "LAYOUT_RESPONSE_FORMAT",
    "LAYOUT_WEIGHTED",
    "LAYOUT_MAX_SAMPLES",
    "JOB_CONFIG_YAML",
    "CACHED_DATASET_S3_PREFIX",
    "CACHED_DATASET_S3_PATHS",
    "CACHED_DATASET_SLICES",
    "CACHED_DATASET_LOCAL_PATHS",
    "CACHED_DATASET_LOCAL_ROOT",
    "SWIFT_RANK_LOCAL_ITERABLE_DATASET",
    "USE_HF",
    "USE_HF_TRANSFER",
    "IO_DUMP_S3_PREFIX",
    "IO_DUMP_MAX_SAMPLES",
    "DEBUG_LOG_S3",
    "WORKLOAD_STARTING_TIMEOUT",
    "WORKLOAD_RECOVERING_TIMEOUT",
    "HF_ARTIFACT_ATTEMPT_ID",
    "INPUT_PARTS_S3",
    "INPUT_PARTITION_INDEX_SHA256",
    "INFERENCE_TOTAL_GPU_COUNT",
)

OPTIONAL_ENV_DEFAULTS = {
    "DLMS_RUN_SCRIPT": "launch_b300.sh",
    "USE_HF": "0",
    "USE_HF_TRANSFER": "0",
    "WORKLOAD_STARTING_TIMEOUT": "10h",
    "WORKLOAD_RECOVERING_TIMEOUT": "2h",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    return doc


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _template(value: Any, mapping: dict[str, str]) -> str:
    return Template(_as_str(value)).safe_substitute(mapping)


def _app_path(repo_root: Path, local_path: Path) -> str:
    try:
        rel = local_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        raise SystemExit(
            f"{local_path} is outside the repo; put job YAML under job_config/ so it is copied into the image"
        )
    return f"/app/{rel.as_posix()}"


def _config_register_path(path: str) -> str:
    if not path:
        path = "configs/config_from_job_yaml.py"
    if path.startswith("/"):
        return path
    return f"/app/{path}"


def _source_jsonls(source: dict[str, Any]) -> list[str]:
    value = source.get("jsonl_paths", source.get("jsonl", ""))
    if isinstance(value, list):
        return [_as_str(v) for v in value if _as_str(v)]
    value = _as_str(value)
    return [value] if value else []


def _derive_layout_env(data_config: dict[str, Any]) -> dict[str, str]:
    sources = data_config.get("sources") or []
    if not sources:
        return {}

    jsonls: list[str] = []
    roots: list[str] = []
    max_samples = "0"
    response_format = ""

    for source in sources:
        if not isinstance(source, dict):
            continue
        source_jsonls = _source_jsonls(source)
        jsonls.extend(source_jsonls)
        roots.extend([_as_str(source.get("images_root", ""))] * len(source_jsonls))
        if not response_format:
            response_format = _as_str(source.get("response_format", ""))
        if max_samples == "0" and _as_str(source.get("max_samples", "0")) != "0":
            max_samples = _as_str(source.get("max_samples", "0"))

    env = {
        "LAYOUT_JSONL_PATHS": ",".join(jsonls),
        "LAYOUT_IMAGES_ROOTS": ",".join(roots),
        "LAYOUT_RESPONSE_FORMAT": response_format or "semicolon_with_type",
        "LAYOUT_WEIGHTED": "1" if data_config.get("use_weighted_sampling") else "0",
        "LAYOUT_MAX_SAMPLES": max_samples,
    }
    return env


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def _resolve_utp_command() -> list[str]:
    """Prefer an installed UTP CLI and fall back to the core-cn skill copy."""
    installed = shutil.which("utp")
    if installed:
        return [installed]
    if UTP_SKILL_FALLBACK.is_file():
        uv = shutil.which("uv")
        if not uv:
            raise SystemExit("The UTP skill fallback requires 'uv' on PATH to provide its dependencies")
        command = [uv, "run", "--isolated", "--no-project"]
        for dependency in UTP_SKILL_DEPENDENCIES:
            command.extend(["--with", dependency])
        return [*command, str(UTP_SKILL_FALLBACK)]
    raise SystemExit(
        f"Could not find 'utp' on PATH and the core-cn skill fallback is missing: {UTP_SKILL_FALLBACK}"
    )


def _validate_raw_jobset_config(repo_root: Path, path: Path, doc: dict[str, Any]) -> str:
    submit_with = doc.get("submit_with") or {}
    template = _as_str(submit_with.get("jobset_template"))
    if not template:
        raise SystemExit(f"{path} missing submit_with.jobset_template")
    template_path = (repo_root / template).resolve()
    try:
        template_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"Raw JobSet template must be inside the repository: {template_path}") from exc
    template_doc = _load_yaml(template_path)
    if template_doc.get("kind") != "JobSet" or not _as_str(template_doc.get("apiVersion")).startswith(
        "jobset.x-k8s.io/"
    ):
        raise SystemExit(f"{template_path} is not a Kubernetes JobSet")
    replicated_jobs = template_doc.get("spec", {}).get("replicatedJobs", [])
    job_names = {_as_str(job.get("name")) for job in replicated_jobs if isinstance(job, dict)}
    if "workers" not in job_names or any("head" in name for name in job_names):
        raise SystemExit(
            f"{template_path} must contain workers and no Ray/head replicated job; found {sorted(job_names)}"
        )
    return template


def _build_submit_command(
    utp_command: list[str],
    *,
    template: str,
    job_name: str,
    dry_run: bool,
    passthrough: list[str],
) -> list[str]:
    cmd = list(utp_command)
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["s", job_name, template])
    if passthrough:
        cmd.extend(["--", *passthrough])
    return cmd


def _build_direct_arnold_command(
    arnold_command: str,
    *,
    template: str,
    job_name: str,
    application_id: str,
    dry_run: bool,
    passthrough: list[str],
) -> list[str]:
    """Build the modern direct-JobSet Arnold submission command."""

    cmd = [
        arnold_command,
        "submit",
        "--filename",
        template,
        "--application-id",
        application_id,
        "--job-name",
        job_name,
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(passthrough)
    return cmd


def _scoped_kubeconfig(context: str) -> str:
    """Return a temporary kubeconfig with `context` selected."""
    try:
        kubeconfig = subprocess.check_output(["kubectl", "config", "view", "--raw"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not read kubeconfig for submit_with.kube_context={context}: {exc}") from exc

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="dlswift-kubeconfig-", delete=False)
    try:
        tmp.write(kubeconfig)
        tmp.close()
        subprocess.run(
            ["kubectl", "--kubeconfig", tmp.name, "config", "use-context", context],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return tmp.name
    except subprocess.CalledProcessError as exc:
        os.unlink(tmp.name)
        raise SystemExit(
            f"Could not select submit_with.kube_context={context}. "
            f"Available contexts can be listed with `kubectl config get-contexts`.\n{exc.stderr}"
        ) from exc


def main() -> int:
    argv, passthrough = _split_passthrough(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="job_config YAML")
    parser.add_argument("job_name", nargs="?", help="Job name suffix. Defaults to YAML name.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Ask utp to print the Arnold command without submitting."
    )
    parser.add_argument(
        "--direct-arnold",
        action="store_true",
        help="Submit the raw JobSet directly with Arnold 3.x instead of the legacy utp wrapper.",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    config_path = (Path.cwd() / args.config).resolve()
    doc = _load_yaml(config_path)

    submit_with = doc.get("submit_with") or {}
    template = _validate_raw_jobset_config(repo_root, config_path, doc)

    yaml_name = _as_str(doc.get("name") or config_path.stem)
    job_name = args.job_name or yaml_name
    owner = os.environ.get("DEVBOX_OWNER") or os.environ.get("USER", "unknown")

    mapping = {**os.environ, "JOB_NAME": job_name, "OWNER": owner}
    child_env = os.environ.copy()
    child_env.update({"JOB_NAME": job_name, "OWNER": owner})

    for key in OPTIONAL_ENV_VARS:
        child_env.setdefault(key, OPTIONAL_ENV_DEFAULTS.get(key, ""))

    data_config = doc.get("data_config") or {}
    if isinstance(data_config, dict):
        for key, value in _derive_layout_env(data_config).items():
            child_env.setdefault(key, value)

    raw_env = {}
    for section in ("env_vars", "env"):
        values = doc.get(section) or {}
        if not isinstance(values, dict):
            raise SystemExit(f"{config_path} field {section} must be a mapping")
        raw_env.update(values)

    compute_config = doc.get("compute_config") or {}
    compute_flags = compute_config.get("flags") if isinstance(compute_config, dict) else {}
    if not isinstance(compute_flags, dict):
        compute_flags = {}
    raw_env.setdefault(
        "WORKLOAD_STARTING_TIMEOUT",
        compute_flags.get("workload_starting_timeout", OPTIONAL_ENV_DEFAULTS["WORKLOAD_STARTING_TIMEOUT"]),
    )
    raw_env.setdefault(
        "WORKLOAD_RECOVERING_TIMEOUT",
        compute_flags.get(
            "workload_recovering_timeout", OPTIONAL_ENV_DEFAULTS["WORKLOAD_RECOVERING_TIMEOUT"]
        ),
    )

    if "max_retries" in doc and "MAX_RETRIES" not in raw_env:
        raw_env["MAX_RETRIES"] = doc.get("max_retries")

    if data_config and "CUSTOM_REGISTER_PATH" not in raw_env:
        raw_env["CUSTOM_REGISTER_PATH"] = _config_register_path(
            _as_str(data_config.get("custom_register_path", "configs/config_from_job_yaml.py"))
        )
    raw_env.setdefault("JOB_CONFIG_YAML", _app_path(repo_root, config_path))

    mapping.update({k: _as_str(v) for k, v in child_env.items()})
    for key, value in raw_env.items():
        child_env[key] = _template(value, mapping)
        mapping[key] = child_env[key]

    extra = list(passthrough)
    priority = submit_with.get("arnold_priority")
    if priority and "--priority" not in extra:
        extra.extend(["--priority", _as_str(priority)])

    dry_run = args.dry_run or os.environ.get("DRY_RUN") == "1"
    if args.direct_arnold:
        arnold_command = shutil.which("arnold")
        if not arnold_command:
            raise SystemExit("Could not find 'arnold' on PATH for --direct-arnold submission")
        cmd = _build_direct_arnold_command(
            arnold_command,
            template=template,
            job_name=job_name,
            application_id=os.environ.get("UTP_APPLICATION_ID", "core-cn"),
            dry_run=dry_run,
            passthrough=extra,
        )
    else:
        cmd = _build_submit_command(
            _resolve_utp_command(),
            template=template,
            job_name=job_name,
            dry_run=dry_run,
            passthrough=extra,
        )

    print(f"[submit_from_job_yaml] config={config_path.relative_to(repo_root)}", flush=True)
    submit_mode = "direct-arnold" if args.direct_arnold else "raw-jobset"
    print(f"[submit_from_job_yaml] job={job_name} mode={submit_mode} template={template}", flush=True)
    print(f"[submit_from_job_yaml] submitter={' '.join(cmd[:1])}", flush=True)
    kube_context = _template(submit_with.get("kube_context", ""), mapping)
    scoped_kubeconfig = ""
    if kube_context:
        scoped_kubeconfig = _scoped_kubeconfig(kube_context)
        child_env["KUBECONFIG"] = scoped_kubeconfig
        print(f"[submit_from_job_yaml] kube_context={kube_context}", flush=True)
    print(
        f"[submit_from_job_yaml] CUSTOM_REGISTER_PATH={child_env.get('CUSTOM_REGISTER_PATH', '')}", flush=True
    )
    print(f"[submit_from_job_yaml] JOB_CONFIG_YAML={child_env.get('JOB_CONFIG_YAML', '')}", flush=True)
    print(
        "[submit_from_job_yaml] "
        f"WORKLOAD_STARTING_TIMEOUT={child_env.get('WORKLOAD_STARTING_TIMEOUT', '')} "
        f"WORKLOAD_RECOVERING_TIMEOUT={child_env.get('WORKLOAD_RECOVERING_TIMEOUT', '')}",
        flush=True,
    )
    try:
        return subprocess.run(cmd, cwd=repo_root, env=child_env).returncode
    finally:
        if scoped_kubeconfig:
            try:
                os.unlink(scoped_kubeconfig)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
