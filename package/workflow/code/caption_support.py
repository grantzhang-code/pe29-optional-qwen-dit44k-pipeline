from pathlib import Path
import hashlib,os
ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'code'
BASE=ROOT/'caption'
ASSETS=ROOT.parent/'assets'
from checkpoint_sources import cache_path, load_sources
MODEL=cache_path('pe29_hf')
VENV=ASSETS/'env/caption/bin/python'
CUDA_HOME=ASSETS/'env/caption/lib/python3.12/site-packages/nvidia/cu13'
STEP=29000
sha=lambda b:hashlib.sha256(b).hexdigest()
def make_command(rank: int, input_path: Path, output_dir: Path, lock: dict, *, node_count: int) -> list[str]:
    if lock['model_uri'] != load_sources()['pe29_hf']:
        raise ValueError('PE model source differs from the model lock')
    data = input_path.read_bytes()
    return [
        str(VENV), "-u", str(TASK / "infer_caption.py"),
        "--model", str(MODEL),
        "--artifact-manifest-sha256", lock["trained_artifact_manifest_sha256"],
        "--checkpoint-step", str(STEP),
        "--checkpoint-manifest-sha256", lock["checkpoint_manifest_sha256"],
        "--input-jsonl", str(input_path), "--input-sha256", sha(data),
        "--expected-count", str(data.count(b"\n")),
        "--output-jsonl", str(output_dir / "predictions.jsonl"),
        "--summary-json", str(output_dir / "summary.json"),
        "--node-index", str(rank), "--node-count", str(node_count),
        "--batch-size", "4", "--request-window-size", "16", "--download-workers", "8",
        "--max-model-len", "32768", "--max-new-tokens", "27000",
        "--max-pixels", "2359296", "--gpu-memory-utilization", "0.90",
        "--attention-backend", "FLASHINFER", "--num-speculative-tokens", "1", "--seed", "42",
    ]

def environment(gpu: int, rank: int) -> dict[str, str]:
    cache = BASE / "cache" / f"rank-{rank:02d}"
    short_tmp = Path(f"/tmp/p29qr-r{rank:02d}")
    for path in (cache, short_tmp, BASE / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu), CUDA_HOME=str(CUDA_HOME),
        PATH=str(CUDA_HOME / "bin") + os.pathsep + os.environ.get("PATH", ""),
        TOKENIZERS_PARALLELISM="false", VLLM_USE_DEEP_GEMM="0",
        VLLM_MOE_USE_DEEP_GEMM="0", VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_NO_USAGE_STATS="1", VLLM_ENABLE_CUDA_COMPATIBILITY="0",
        TORCH_CUDA_ARCH_LIST="9.0", PYTORCH_ALLOC_CONF="expandable_segments:True",
        VLLM_CACHE_ROOT=str(cache), HF_HOME=str(cache / "hf"), XDG_CACHE_HOME=str(cache),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"),
        TMPDIR=str(short_tmp),
    )
