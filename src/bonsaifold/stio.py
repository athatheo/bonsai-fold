"""Safetensors IO and MLX affine-quantization primitives.

Deliberately independent of mlx: these helpers are the *cross-check* for
mx.dequantize and the substrate for format-preserving surgery, so they parse
the bytes themselves. A future pack writer copies raw buffers (read_raw) for
surviving tensors and only encodes merged ones.
"""
import json
import struct
from pathlib import Path

import numpy as np

DTYPE_NP = {"F16": np.float16, "BF16": np.uint16, "F32": np.float32, "U32": np.uint32}


def read_st_header(path):
    """Return (header dict without __metadata__, data start offset)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def read_tensor(path, header, data_start, name):
    info = header[name]
    begin, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        raw = f.read(end - begin)
    arr = np.frombuffer(raw, dtype=DTYPE_NP[info["dtype"]]).reshape(info["shape"])
    if info["dtype"] == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


def weight_base(name):
    """Base name of a weight tensor (sans .weight); None if not a weight."""
    return name[: -len(".weight")] if name.endswith(".weight") else None


def is_quantized(name, header):
    """True if `name` is an MLX affine-quantized weight in this header."""
    base = weight_base(name)
    return (
        base is not None
        and header[name]["dtype"] == "U32"
        and base + ".scales" in header
    )


class PackReader:
    """Uniform reader over an MLX model pack, single-shard or sharded.

    Resolves tensors through model.safetensors.index.json when present, so the
    same code reads shipped packs and future folded packs saved by mlx-lm
    (which shards at 5 GB by default).
    """

    def __init__(self, pack_dir):
        self.pack_dir = Path(pack_dir)
        self.config = json.loads((self.pack_dir / "config.json").read_text())
        index = self.pack_dir / "model.safetensors.index.json"
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.header = {}  # tensor name -> info dict
        self._src = {}  # tensor name -> (path, per-shard header, data_start)
        for fname in files:
            path = self.pack_dir / fname
            header, data_start = read_st_header(path)
            for name, info in header.items():
                self.header[name] = info
                self._src[name] = (path, header, data_start)

    def read(self, name):
        return read_tensor(*self._src[name], name)

    def read_raw(self, name):
        """The tensor's exact stored bytes (for byte-identical pass-through)."""
        path, header, data_start = self._src[name]
        begin, end = header[name]["data_offsets"]
        with open(path, "rb") as f:
            f.seek(data_start + begin)
            return f.read(end - begin)

    def quant_params(self, tensor_base=None):
        """(group_size, bits) from config for a tensor (``base`` = name sans
        .weight), honoring per-tensor overrides mlx-lm may write."""
        q = self.config.get("quantization", {})
        gs, bits = q.get("group_size"), q.get("bits")
        if tensor_base is not None and isinstance(q.get(tensor_base), dict):
            over = q[tensor_base]
            gs, bits = over.get("group_size", gs), over.get("bits", bits)
        return gs, bits

    def quant_meta(self, weight_name):
        """(group_size, bits) for a quantized weight tensor, with bits
        inferred empirically from header shapes (config bits is a hint only)."""
        base = weight_base(weight_name)
        gs, _ = self.quant_params(base)
        bits = infer_bits(
            self.header[weight_name]["shape"], self.header[base + ".scales"]["shape"], gs
        )
        return gs, bits


def block_owner(name):
    """Map a tensor name to its owner: a language-block index (int),
    'vision_tower', or 'tail' (embeddings / lm_head / final norm)."""
    if name.startswith("vision_tower."):
        return "vision_tower"
    parts = name.split(".")
    if "layers" in parts:
        return int(parts[parts.index("layers") + 1])
    return "tail"


def reindex_name(name, block_map):
    """Rewrite a tensor (or config-key) name under a block renumbering.

    block_map: old block index -> new block index. Non-layer names pass
    through unchanged; a name whose block is not in the map (a dropped
    block) returns None.
    """
    parts = name.split(".")
    if "layers" not in parts or name.startswith("vision_tower."):
        return name
    i = parts.index("layers") + 1
    new = block_map.get(int(parts[i]))
    if new is None:
        return None
    return ".".join(parts[:i] + [str(new)] + parts[i + 1 :])


def write_safetensors(path, entries):
    """Write a safetensors file from raw buffers.

    entries: name -> {"dtype": st_dtype, "shape": [...], "nbytes": int,
    "raw": bytes | () -> bytes}. Buffers are written verbatim
    (byte-identical pass-through for surgery). A callable `raw` is invoked
    one tensor at a time, so a whole pack never sits in memory; "nbytes"
    may be omitted when `raw` is bytes.
    """
    header = {}
    offset = 0
    names = list(entries)
    for name in names:
        e = entries[name]
        size = e["nbytes"] if "nbytes" in e else len(e["raw"])
        header[name] = {
            "dtype": e["dtype"],
            "shape": list(e["shape"]),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)  # 8-byte alignment, spec-conventional
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in names:
            raw = entries[name]["raw"]
            data = raw() if callable(raw) else raw
            if len(data) != header[name]["data_offsets"][1] - header[name]["data_offsets"][0]:
                raise ValueError(f"{name}: payload size != declared nbytes")
            f.write(data)


def unpack_codes(packed_u32, bits):
    """Unpack MLX-packed quantization codes (LSB-first within each u32)."""
    per_word = 32 // bits
    mask = (1 << bits) - 1
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    codes = (packed_u32[..., None] >> shifts) & mask
    return codes.reshape(*packed_u32.shape[:-1], -1)


def infer_bits(weight_shape, scales_shape, group_size):
    """Bits per weight from a packed U32 weight [rows, packed_cols] and its
    scales [rows, cols/group_size]."""
    cols = scales_shape[-1] * group_size
    return 32 * weight_shape[-1] // cols


def manual_dequant(packed_u32, scales, biases, bits, group_size):
    """Affine dequantization w = s*q + b, independent of mlx."""
    codes = unpack_codes(packed_u32, bits).astype(np.float32)
    s = scales.astype(np.float32).repeat(group_size, axis=-1)
    b = biases.astype(np.float32).repeat(group_size, axis=-1)
    return codes * s + b
