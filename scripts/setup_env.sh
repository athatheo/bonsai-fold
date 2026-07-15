#!/usr/bin/env bash
# Bootstrap the bonsai-fold environment from a fresh checkout.
#
# 1-bit MLX support is not in stock mlx (pending ml-explore/mlx#3161); we
# build PrismML's fork (branch `prism`) from source, pinned to the commit the
# ABI audit below was performed against.
#
# This machine has Xcode 26.3 without the Metal Toolchain component, and
# installing it needs admin (xcodebuild -runFirstLaunch hangs on an auth
# prompt). We therefore build WITHOUT the Metal offline compiler:
#   - MLX_METAL_JIT=ON: all fork-modified kernels (quantized incl. 1-bit,
#     steel gemm, ...) are embedded as source and JIT-compiled at runtime.
#   - MLX_PREBUILT_METALLIB=...: the always-offline base kernels (arg_reduce,
#     conv, gemv, layer_norm, random, rms_norm, rope, sdpa, fence) are taken
#     from the stock mlx-metal 0.31.2 wheel's mlx.metallib. Audited at
#     MLX_FORK_COMMIT below: the fork does not change any of these kernels'
#     host<->kernel ABI (random.metal's ulong->uint narrowing is kernel-side
#     only and the stock pairing is the original consistent one; spec_decode
#     is fork-new, absent from the stock metallib, and only reachable via
#     speculative decoding, which we never use; NAX kernels are M5-only and
#     this is an M4 Max). Re-audit before bumping the pin.
#   - make_compiled_preamble.sh's `xcrun metal -E -H` include walk is
#     replaced by scripts/jit_preamble_deps.py (pure python).
#
# Local fork modifications = patches/mlx-prism-no-metal-toolchain.patch plus
# the copied-in walker (single source: scripts/jit_preamble_deps.py; the copy
# below runs every time, so walker edits propagate on re-run). Regenerate the
# patch after changing the fork tree: git -C vendor/mlx diff > patches/...
# Validation gate: experiments/runtime_validation/validate_runtime.py
#
# If a Metal Toolchain becomes available (admin required):
#   sudo xcodebuild -runFirstLaunch
#   xcodebuild -downloadComponent MetalToolchain
# then rebuild without MLX_PREBUILT_METALLIB for a fully offline-compiled
# runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

MLX_FORK_COMMIT=10b5fe43b94a4ed4eb941fb3b74723af7dfbe8b4
export CMAKE_ARGS="-DMLX_METAL_JIT=ON -DMLX_PREBUILT_METALLIB=$PWD/vendor/prebuilt/mlx.metallib"

# Clone / pin / patch idempotently: every step re-checked on every run, so an
# interrupted first run or an updated patch cannot leave a silently stale tree.
if [ ! -d vendor/mlx ]; then
  git clone -b prism https://github.com/PrismML-Eng/mlx.git vendor/mlx
fi
if [ "$(git -C vendor/mlx rev-parse HEAD)" != "$MLX_FORK_COMMIT" ]; then
  git -C vendor/mlx fetch origin "$MLX_FORK_COMMIT"
  git -C vendor/mlx checkout "$MLX_FORK_COMMIT"
fi
if ! git -C vendor/mlx apply --reverse --check ../../patches/mlx-prism-no-metal-toolchain.patch 2>/dev/null; then
  git -C vendor/mlx apply ../../patches/mlx-prism-no-metal-toolchain.patch
fi
cp scripts/jit_preamble_deps.py vendor/mlx/mlx/backend/metal/jit_preamble_deps.py

if [ ! -f vendor/prebuilt/mlx.metallib ]; then
  mkdir -p vendor/prebuilt
  tmp=$(mktemp -d)
  curl -sL "https://files.pythonhosted.org/packages/99/82/11fd62a8d7a3e96e5c43220b17de0151e3f10101f8bb3b865f5bd9cdd074/mlx_metal-0.31.2-py3-none-macosx_26_0_arm64.whl" \
    -o "$tmp/w.whl"
  unzip -o -q "$tmp/w.whl" -d "$tmp"
  cp "$tmp/mlx/lib/mlx.metallib" vendor/prebuilt/mlx.metallib
  rm -rf "$tmp"
fi

# pyproject's [tool.uv.sources] points mlx at vendor/mlx (editable), so uv
# owns the install; CMAKE_ARGS above is required whenever uv (re)builds it.
# Two-step: mlx builds with no build isolation, so its build deps (setuptools,
# cmake — the dev group) must land in the venv BEFORE mlx builds. The
# --reinstall-package on the second step also forces a rebuild: uv's cache key
# only watches vendor/mlx's pyproject/setup files, not the kernel sources or
# our patch, so an unconditional rebuild here is the correctness guarantee.
uv sync --no-install-package mlx
uv sync --reinstall-package mlx

uv run python experiments/runtime_validation/validate_runtime.py
