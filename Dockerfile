# syntax=docker/dockerfile:1
#
# Reproduces the local `navsim_gpu` environment (Python 3.9, torch 2.7.1+cu128,
# CUDA 12.8) so a remote host can run this repository without a conda install.
#
# Why `pip install -r requirements.txt` alone is not enough:
#   requirements.txt declares 16 packages while the working environment has 243.
#   It also omits several modules the code imports directly (matplotlib, pyarrow,
#   pandas, psutil), and it pins `torch==2.7.1` without a wheel index, which
#   resolves to the CPU build from PyPI. This file installs the requirements
#   first, then force-reinstalls torch from the cu128 index, then adds the
#   missing direct dependencies by version observed on the reference machine.
#
# Deliberately not baked into the image:
#   outputs/   feature indices, checkpoints and model assets -> mount a volume
#   .git       both the repository and the submodule history (7 MB) are unused
#   data/      raw datasets live on the host

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# System layer
#   libgl1 / libglib2.0-0 / libsm6 / libxext6  -> cv2 and PIL import at runtime
#   git                                        -> nuplan-devkit is a git URL
#   software-properties-common                 -> deadsnakes PPA for Python 3.9
#
# Ubuntu 22.04 ships Python 3.10.  The reference environment is 3.9.23 and the
# whole 243-package graph is already resolved against that ABI, so the image
# keeps 3.9 rather than silently re-resolving every wheel for 3.10.  The cu128
# index does publish cp39 wheels for torch 2.7.1 and torchvision 0.22.1
# (manylinux_2_28, glibc 2.35 on jammy is new enough).
#
# This is the *runtime* CUDA image (no nvcc).  Nothing here compiles CUDA
# extensions; if that changes, switch to the -devel tag.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg2 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-dev" \
        "python${PYTHON_VERSION}-venv" \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | "python${PYTHON_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# A venv keeps the interpreter isolated from the distro packages and matches the
# conda layout the project was developed against (no --user installs).
ENV VIRTUAL_ENV=/opt/venv
RUN "python${PYTHON_VERSION}" -m venv "$VIRTUAL_ENV"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /tmp/build

# ---------------------------------------------------------------------------
# Python layer
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/build/requirements.txt

# Torch first from the cu128 index so the heavyweight wheel is resolved against
# CUDA 12.8 instead of PyPI's CPU build; --no-deps because requirements.txt
# installs the rest of the graph immediately afterwards.
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX}" --no-deps \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"

# The requirements pins torch again. --no-deps combined with a satisfied
# requirement keeps the cu128 wheel, but pip would still try to swap it for the
# PyPI build, so the torch lines are filtered out before installing.
RUN grep -viE '^[[:space:]]*(torch|torchvision)[[:space:]]*==' /tmp/build/requirements.txt \
        > /tmp/build/requirements.no-torch.txt \
    && pip install --no-cache-dir -r /tmp/build/requirements.no-torch.txt

# Direct imports that requirements.txt does not declare. Versions are the ones
# observed in the reference environment; they are not resolver-chosen.
RUN pip install --no-cache-dir \
        "matplotlib==3.9.4" \
        "pandas==2.3.3" \
        "pyarrow==21.0.0" \
        "psutil==7.2.2"

# ---------------------------------------------------------------------------
# Source layer
#   `navsim` and the model code live in the official submodule, so the project
#   itself is only importable with PYTHONPATH pointing at it.
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY repositories/DiffusionDrive/ /workspace/repositories/DiffusionDrive/
COPY agent/ /workspace/agent/
COPY tools/ /workspace/tools/
COPY tests/ /workspace/tests/
COPY README.md requirements.txt /workspace/

ENV PYTHONPATH=/workspace/repositories/DiffusionDrive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg

# Build-time smoke test: fails the build instead of failing at first use. A single
# `import` statement is used instead of a here-document so the file also builds on
# the legacy (non-BuildKit) builder; an ImportError names the missing module.
# This needs no GPU.
RUN python -c "import torch, torchvision, pytorch_lightning, numpy, scipy, matplotlib, shapely, cv2, PIL, pandas, pyarrow, psutil, timm, diffusers, hydra, omegaconf, navsim, nuplan; print('imports OK | torch', torch.__version__, '| cuda', torch.version.cuda)"

# Dataset roots, feature stores and checkpoints are host state, not image state.
# Mount them at run time, e.g.:
#   -v /host/bench2drive:/data/datasets/Bench2Drive-V0.0.4/v1:ro
#   -v /host/features:/data/diffusiondrive_v1_tensors/features:ro
#   -v /host/experiments:/run/media/aethelind/PSSD/diffusiondrive_v1_experiments
VOLUME ["/data/datasets", "/data/diffusiondrive_v1_tensors", "/run/media/aethelind/PSSD"]

WORKDIR /workspace

# Training and evaluation both enter through the same module entry points; the
# default command is a CPU dry run, which validates the data index without
# touching a GPU.
CMD ["python", "-m", "tools.training.diffusiondrive_v1.status_experimental_train", "--help"]
