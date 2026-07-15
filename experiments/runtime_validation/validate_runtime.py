"""Validate the hybrid MLX build (PrismML fork, JIT kernels + borrowed
stock-0.31.2 metallib for base kernels) before trusting any experiment.

Checks:
  1. GPU == CPU for every borrowed-metallib kernel we rely on
     (random, rms_norm, layer_norm, rope, sdpa, gemv, arg_reduce, conv).
  2. Affine quantize/dequantize and quantized_matmul at bits in {1, 2}
     (1-bit = primary arm, 2-bit = ternary arm), GPU vs CPU vs manual numpy.
  3. Real pack tensor: mx.dequantize vs independent numpy unpacking, plus the
     format's value invariants (1-bit: values exactly +/- s_g up to f16
     subnormal rounding; 2-bit: biases == -scales, code 3 unused,
     values in {-s_g, 0, +s_g}).

Usage: uv run python experiments/runtime_validation/validate_runtime.py \
           [pack_dir]   (default: <repo>/models/Bonsai-27B-mlx-1bit)
A missing pack skips check 3 with a loud warning (the kernel checks gate
independently — setup_env.sh runs on machines without the 5 GB pack).
Exits nonzero on any failure. Results JSON written next to this file.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsaifold.stio import PackReader, is_quantized, manual_dequant, unpack_codes, weight_base

REPO_ROOT = Path(__file__).resolve().parents[2]
F16_SUBNORMAL_ATOL = 6.1e-08  # 1 ulp at the bottom of the f16 subnormal range
RESULTS = {"mlx_version": mx.__version__, "checks": []}


def check(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def check_close(name, a, b, atol):
    check(name, np.allclose(a, b, atol=atol), f"max|d|={np.abs(a - b).max():.2e}")


def check_gpu_cpu(name, fn, atol=None):
    """Run fn(stream) on gpu and cpu; atol=None means require exact equality."""
    outs = []
    for dev in (mx.gpu, mx.cpu):
        out = fn(dev)
        mx.eval(out)
        outs.append(np.array(out, copy=False).astype(np.float64))
    g, c = outs
    if atol is None:
        check(f"{name} gpu==cpu", np.array_equal(g, c))
    else:
        check_close(f"{name} gpu~cpu", g, c, atol)


def base_kernels():
    mx.random.seed(0)
    x = mx.random.normal((8, 512))
    w = mx.random.normal((512,))
    b = mx.random.normal((512,))
    xr = mx.random.normal((1, 4, 16, 128))
    q = mx.random.normal((1, 4, 1, 128))
    k = mx.random.normal((1, 1, 64, 128))
    v = mx.random.normal((1, 1, 64, 128))
    m = mx.random.normal((256, 512))
    vec = mx.random.normal((1, 512))
    xc = mx.random.normal((2, 32, 8))
    wc = mx.random.normal((4, 3, 8))
    key = mx.array([12, 34], dtype=mx.uint32)

    check_gpu_cpu("random.uniform", lambda d: mx.random.uniform(shape=(64, 128), key=key, stream=d))
    check_gpu_cpu("rms_norm", lambda d: mx.fast.rms_norm(x, w, 1e-5, stream=d), 1e-2)
    check_gpu_cpu("layer_norm", lambda d: mx.fast.layer_norm(x, w, b, 1e-5, stream=d), 1e-2)
    check_gpu_cpu(
        "rope",
        lambda d: mx.fast.rope(xr, 32, traditional=False, base=10000.0, scale=1.0, offset=7, stream=d),
        1e-3,
    )
    check_gpu_cpu(
        "sdpa(vector)",
        lambda d: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.088, mask=None, stream=d),
        1e-3,
    )
    check_gpu_cpu("gemv", lambda d: mx.matmul(vec, mx.transpose(m), stream=d), 1e-2)
    check_gpu_cpu("arg_reduce", lambda d: mx.argmax(x, axis=-1, stream=d))
    check_gpu_cpu("conv1d", lambda d: mx.conv1d(xc, wc, stream=d), 1e-2)


def quant_ops(bits, gs=128):
    tag = f"{bits}-bit"
    mx.random.seed(bits)
    w = mx.random.normal((256, 512))
    q, s, b = mx.quantize(w, group_size=gs, bits=bits, stream=mx.cpu)
    mx.eval(q, s, b)
    manual = manual_dequant(
        np.array(q, copy=False), np.array(s, copy=False), np.array(b, copy=False), bits, gs
    )
    for dev, dtag in ((mx.gpu, "gpu"), (mx.cpu, "cpu")):
        d = mx.dequantize(q, s, b, group_size=gs, bits=bits, stream=dev)
        mx.eval(d)
        check_close(
            f"{tag} dequantize == manual unpack ({dtag})", np.array(d, copy=False), manual, 1e-3
        )

    # GPU and CPU quantize may break exact-midpoint ties differently (codes
    # differ on a handful of elements) — benign, and we never mx.quantize
    # model weights. Require identical scales/biases and identical
    # reconstruction error instead of bitwise-equal codes.
    qg, sg, bg = mx.quantize(w, group_size=gs, bits=bits, stream=mx.gpu)
    mx.eval(qg, sg, bg)
    wv = np.array(w, copy=False).astype(np.float64)
    err = {}
    for etag, (qq, ss, bb) in (("cpu", (q, s, b)), ("gpu", (qg, sg, bg))):
        d = mx.dequantize(qq, ss, bb, group_size=gs, bits=bits, stream=mx.cpu)
        mx.eval(d)
        err[etag] = float(np.abs(np.array(d, copy=False).astype(np.float64) - wv).mean())
    check(
        f"{tag} quantize gpu~cpu (same scales/biases, same recon error)",
        np.array_equal(np.array(sg, copy=False), np.array(s, copy=False))
        and np.array_equal(np.array(bg, copy=False), np.array(b, copy=False))
        and abs(err["cpu"] - err["gpu"]) < 1e-9,
        f"recon err cpu={err['cpu']:.6f} gpu={err['gpu']:.6f}",
    )

    x = mx.random.normal((4, 512))
    deq = mx.dequantize(q, s, b, group_size=gs, bits=bits, stream=mx.cpu)
    ref = np.array(mx.matmul(x, mx.transpose(deq), stream=mx.cpu), copy=False)
    for dev, dtag in ((mx.gpu, "gpu"), (mx.cpu, "cpu")):
        y = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=gs, bits=bits, stream=dev)
        mx.eval(y)
        check_close(
            f"{tag} quantized_matmul ~ dequant matmul ({dtag})", np.array(y, copy=False), ref, 2e-2
        )


def pack_tensor(pack_dir):
    pack = PackReader(pack_dir)
    name = next(
        (
            n
            for n in sorted(pack.header)
            if n.endswith("self_attn.q_proj.weight") and is_quantized(n, pack.header)
        ),
        None,
    )
    if name is None:
        check("pack tensor cross-check", False, f"no quantized q_proj tensor in {pack_dir}")
        return
    base = weight_base(name)
    gs, bits = pack.quant_meta(name)
    w, s, b = pack.read(name), pack.read(base + ".scales"), pack.read(base + ".biases")
    manual = manual_dequant(w, s, b, bits, gs)
    mxd = mx.dequantize(mx.array(w), mx.array(s), mx.array(b), group_size=gs, bits=bits, stream=mx.gpu)
    mx.eval(mxd)
    short = name.split(".layers.")[-1]
    check_close(
        f"pack {short} ({bits}-bit): mx.dequantize == numpy unpack",
        np.array(mxd, copy=False).astype(np.float32),
        manual,
        1e-3,
    )
    sf = s.astype(np.float32)
    if bits == 1:
        # |w| == s_g exactly, except 1-ulp f16 rounding of b = f16(-s/2) in
        # the subnormal range (same tolerance as the census).
        check(
            "pack tensor (1-bit): dequant values are +/- s_g (up to f16-subnormal ulp)",
            np.allclose(
                np.abs(manual), (sf / 2).repeat(gs, axis=1), rtol=0, atol=F16_SUBNORMAL_ATOL
            ),
        )
    elif bits == 2:
        check("pack tensor (2-bit): biases == -scales exactly", bool((-s == b).all()))
        odd_mask = np.uint32(0x55555555)
        check(
            "pack tensor (2-bit): code 3 unused",
            not bool((w & (w >> np.uint32(1)) & odd_mask).any()),
        )
        trits = unpack_codes(w, 2).astype(np.float32) - 1.0
        check(
            "pack tensor (2-bit): dequant values in {-s_g, 0, +s_g}",
            np.allclose(manual, trits * sf.repeat(gs, axis=1), atol=1e-7),
        )


if __name__ == "__main__":
    pack_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "models/Bonsai-27B-mlx-1bit"
    base_kernels()
    for bits in (1, 2):
        quant_ops(bits)
    if (pack_dir / "config.json").exists():
        pack_tensor(pack_dir)
    else:
        print(f"WARNING: pack missing ({pack_dir}); skipping pack cross-check")
        RESULTS["pack_check_skipped"] = str(pack_dir)
    out = Path(__file__).parent / "validation_results.json"
    out.write_text(json.dumps(RESULTS, indent=1))
    failed = any(not c["ok"] for c in RESULTS["checks"])
    print(("SOME CHECKS FAILED" if failed else "ALL CHECKS PASSED"), "->", out)
    sys.exit(1 if failed else 0)
