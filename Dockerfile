# syntax=docker/dockerfile:1

# Von container image.
#
# Two variants differ in which PyTorch wheel is installed and whether OpenVINO
# (the `intel` extra) is present:
#   docker build --build-arg TORCH_BACKEND=cpu     -t von:cpu  .   # torch CPU wheel + OpenVINO; `auto` -> openvino:cpu
#   docker build --build-arg TORCH_BACKEND=default -t von:cuda .   # CUDA torch wheel; `auto` -> cuda
#
# The CPU image ships OpenVINO on purpose: Von's measured CPU latency (0.10 s p50
# on c7i.xlarge) is the OpenVINO path; plain torch CPU is several times slower.
#
# Model weights (~3 GB) are NOT baked in; they are fetched from the Hugging
# Face Hub into HF_HOME on first use. Mount a volume there to persist them.

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------- builder ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG PYTHON_VERSION
ARG TORCH_BACKEND=cpu

# Copy the uv binary from the official distroless image rather than running the
# curl installer: this pins the toolchain version and removes the apt layer.
# The binary is statically linked, so it runs on this glibc base.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Dependency layer first, so later source edits do not invalidate the torch install.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${TORCH_BACKEND}" = "cpu" ]; then EXTRA="--extra intel"; else EXTRA=""; fi \
 && uv export --locked --no-dev --no-emit-project --no-hashes ${EXTRA} \
      --format requirements-txt -o /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python "${PYTHON_VERSION}" \
 && if [ "${TORCH_BACKEND}" = "cpu" ]; then \
      # uv.lock pins the CUDA-13 PyTorch wheel, whose ~19 nvidia-*/cuda-*/triton
      # dependencies are emitted as top-level pins in the export. The CPU wheel
      # needs none of them, so drop those lines before installing.
      # --no-hashes is required because the CPU wheel's digest differs from the
      # CUDA wheel recorded in the lock; versions stay exactly pinned.
      # --torch-backend is a `uv pip`-only feature; `uv sync` cannot express it.
      grep -vE '^(nvidia-|cuda[-_]|triton)' /tmp/requirements.txt > /tmp/requirements.cpu.txt; \
      uv pip install --python /opt/venv --torch-backend=cpu -r /tmp/requirements.cpu.txt; \
    else \
      uv pip install --python /opt/venv -r /tmp/requirements.txt; \
    fi

COPY src ./src
# --no-deps keeps the already-installed torch (possibly the CPU wheel) in place;
# without it, installing von-sdk would re-resolve torch from PyPI.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv --no-deps .

# --------------------------------------------------------------- runtime ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG PYTHON_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/data/huggingface

COPY --from=builder /opt/venv /opt/venv

RUN groupadd --gid 1001 von \
 && useradd --uid 1001 --gid 1001 --shell /usr/sbin/nologin --create-home von \
 && mkdir -p "${HF_HOME}" \
 && chown -R von:von /data

USER von
WORKDIR /home/von

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# No --model flag on purpose: `von serve` defaults to the current release alias,
# so omitting it avoids pinning a version that drifts on upgrade. Arguments after
# the image name are appended to `von serve` (e.g. `--on-overflow refuse`).
ENTRYPOINT ["von", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
