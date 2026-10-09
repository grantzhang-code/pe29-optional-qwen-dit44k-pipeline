#!/usr/bin/env bash
# Source inside a held H100 Pod after unpack_editor_image.py has completed.
set -a
EDITOR_ROOTFS=/mnt/ephemeral/pcb-v2-9c4d-editor-runtime/image-rootfs
EDITOR_BASE=/mnt/ephemeral/pcb-v2-9c4d-editor-runtime
CUDA_HOME="$EDITOR_ROOTFS/usr/local/cuda-13.3"
PATH="$EDITOR_ROOTFS/opt/runtime/bin:$CUDA_HOME/bin:$PATH"
PYTHONPATH="$EDITOR_ROOTFS/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages:/opt/magi-attention-src:/app/vendor:/app:/app/tayo_protocol/code/handoff:/opt/inference:/opt/pixelcraft:/opt/pixelcraft/tasks/pixelcraft:$EDITOR_ROOTFS/opt/runtime/lib/python3.12/site-packages:$EDITOR_ROOTFS/usr/local/lib/python3.12/dist-packages:$EDITOR_BASE/host-site-filtered"
LD_LIBRARY_PATH="$EDITOR_ROOTFS/usr/lib/x86_64-linux-gnu:$EDITOR_ROOTFS/usr/lib/x86_64-linux-gnu/nvshmem/13:$EDITOR_ROOTFS/usr/local/lib:$EDITOR_ROOTFS/usr/local/lib/python3.12/dist-packages/torch/lib:$EDITOR_ROOTFS/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/compat/lib.real${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
TRITON_CUOBJDUMP_PATH="$CUDA_HOME/bin/cuobjdump"
TRITON_NVDISASM_PATH="$CUDA_HOME/bin/nvdisasm"
TRITON_CUDACRT_PATH="$CUDA_HOME/include"
TRITON_CUDART_PATH="$CUDA_HOME/include"
TRITON_CUPTI_LIB_PATH="$CUDA_HOME/lib64"
TRITON_CUPTI_INCLUDE_PATH="$CUDA_HOME/include"
TORCH_CUDA_ARCH_LIST=9.0
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
MAGI_ATTENTION_FA4_BACKEND=1
PIXELCRAFT_ATTN_BACKEND=ffa
PIXELCRAFT_FA4_DIRECT=1
MARIANA_FA4_DIRECT=1
PIXELCRAFT_FA4_EXPAND_GQA=0
PIXELCRAFT_FA4_ISOLATE_SEGMENTS=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
WANDB_MODE=disabled
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
XDG_CACHE_HOME="$EDITOR_BASE/cache"
HF_HOME="$EDITOR_BASE/cache/hf"
TORCH_HOME="$EDITOR_BASE/cache/torch"
TRITON_CACHE_DIR="$EDITOR_BASE/cache/triton"
TORCHINDUCTOR_CACHE_DIR="$EDITOR_BASE/cache/inductor"
CUDA_CACHE_PATH="$EDITOR_BASE/cache/cuda"
TMPDIR="$EDITOR_BASE/cache/tmp"
MAGI_ATTENTION_WORKSPACE_BASE="$EDITOR_BASE/cache/magi"
QUACK_CACHE_DIR="$EDITOR_BASE/cache/quack"
set +a
