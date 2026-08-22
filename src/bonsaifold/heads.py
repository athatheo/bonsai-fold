"""Q5: KV-group head dropping for full-attention blocks (27B hybrid).

The byte-identical unit is a GQA group: 6 query heads (each a contiguous
512-row q_proj slab: 256 query + 256 gate rows) + 1 KV head (256 rows of
k_proj/v_proj) + the matching 1536 o_proj input columns (12 g128 groups —
column cuts stay group-aligned). q_norm/k_norm are per-head_dim (shared),
RoPE is per-head_dim: neither carries per-head parameters, so slicing the
four projections is the whole operation.

Zero-copy screening: HeadGroupDropAdapter wraps a block's live Attention,
holding sliced (lazy) views of its quantized planes; install with
install_head_drop(model, block, groups) and RESTORE the returned original
after screening. Requires derived-kernel modules (no biases plane).
"""
import mlx.core as mx
import mlx_lm.models.qwen3_5 as q35
from mlx_lm.models.base import scaled_dot_product_attention


class _QSlice:
    """Minimal quantized-linear over row/col-sliced planes of a source
    QuantizedLinear (derived-kernel: weight+scales only)."""

    def __init__(self, src, rows=None, packed_cols=None, scale_cols=None):
        w, s = src["weight"], src["scales"]
        if rows is not None:
            w, s = w[rows], s[rows]
        if packed_cols is not None:
            w, s = w[:, packed_cols], s[:, scale_cols]
        self.weight, self.scales = w, s
        self.group_size, self.bits, self.mode = src.group_size, src.bits, src.mode

    def __call__(self, x):
        return mx.quantized_matmul(
            x, self.weight, scales=self.scales, biases=None, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
        )


def _keep_rows(n_units, unit_rows, drop_units):
    keep = [u for u in range(n_units) if u not in set(drop_units)]
    idx = []
    for u in keep:
        idx.extend(range(u * unit_rows, (u + 1) * unit_rows))
    return mx.array(idx)


class HeadGroupDropAdapter:
    """Attention forward with whole GQA groups removed, mirroring
    Qwen3NextAttention.__call__ with adjusted head counts."""

    def __init__(self, attn, drop_groups):
        n_kv = attn.num_key_value_heads
        per = attn.num_attention_heads // n_kv  # Q heads per KV group
        drop_groups = sorted(set(drop_groups))
        if not all(0 <= g < n_kv for g in drop_groups):
            raise ValueError(f"groups must be in 0..{n_kv-1}, got {drop_groups}")
        if len(drop_groups) >= n_kv:
            raise ValueError("cannot drop every KV group")
        hd = attn.head_dim
        self.num_key_value_heads = n_kv - len(drop_groups)
        self.num_attention_heads = self.num_key_value_heads * per
        self.head_dim = hd
        self.scale = attn.scale
        self.q_norm, self.k_norm, self.rope = attn.q_norm, attn.k_norm, attn.rope

        # q_proj: per Q head a 512-row slab (query 256 + gate 256); group g
        # owns heads [g*per, (g+1)*per)
        drop_q_heads = [g * per + i for g in drop_groups for i in range(per)]
        self.q_proj = _QSlice(
            attn.q_proj, rows=_keep_rows(attn.num_attention_heads, hd * 2, drop_q_heads)
        )
        self.k_proj = _QSlice(attn.k_proj, rows=_keep_rows(n_kv, hd, drop_groups))
        self.v_proj = _QSlice(attn.v_proj, rows=_keep_rows(n_kv, hd, drop_groups))
        # o_proj: drop input columns of the dropped Q heads; packed u32 cols
        # = K/32, scale cols = K/128 per head (hd*per multiples of 128)
        keep_packed, keep_scale = [], []
        for h in range(attn.num_attention_heads):
            if h in set(drop_q_heads):
                continue
            keep_packed.extend(range(h * hd // 32, (h + 1) * hd // 32))
            keep_scale.extend(range(h * hd // 128, (h + 1) * hd // 128))
        self.o_proj = _QSlice(
            attn.o_proj, packed_cols=mx.array(keep_packed),
            scale_cols=mx.array(keep_scale),
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q_out = self.q_proj(x)
        queries, gate = mx.split(
            q_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))


def install_head_drop(model, block, groups):
    """Swap block's self_attn for the adapter; returns the original for
    restoration. Block must be a full-attention block."""
    layer = model.language_model.model.layers[block]
    if layer.is_linear:
        raise ValueError(f"block {block} is linear-attention; KV-group drop "
                         "applies to full-attention blocks only")
    original = layer.self_attn
    layer.self_attn = HeadGroupDropAdapter(original, groups)
    return original


def restore_head_drop(model, block, original):
    model.language_model.model.layers[block].self_attn = original
