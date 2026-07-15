# Census: Bonsai-27B-mlx-1bit

- model_type `qwen3_5`, 64 language blocks, quant {'group_size': 128, 'bits': 1}
- full-attention blocks (16): [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
- pattern: `LLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLFLLLF`
- NOTE: mlx-lm's qwen3_5 runtime derives block types POSITIONALLY from `full_attention_interval` (is_linear = (i+1) % 4 != 0); it never reads `layer_types`. Any block-drop harness must re-establish types itself.
- language bytes 4.207 GB, vision tower 0.921 GB (333 tensors, prefix vision_tower.*)
- packing verification: all_ok=True over 498 quantized tensors (0 with no known relation for their bit-width)

| block | type | tensors | params (M) | MB |
|---|---|---|---|---|
| 0 | linear | 30 | 389.3 | 60.0 |
| 1 | linear | 30 | 389.3 | 60.0 |
| 2 | linear | 30 | 389.3 | 60.0 |
| 3 | full | 25 | 378.1 | 58.2 |
| 4 | linear | 30 | 389.3 | 60.0 |
| 5 | linear | 30 | 389.3 | 60.0 |
| 6 | linear | 30 | 389.3 | 60.0 |
| 7 | full | 25 | 378.1 | 58.2 |
| 8 | linear | 30 | 389.3 | 60.0 |
| 9 | linear | 30 | 389.3 | 60.0 |
| 10 | linear | 30 | 389.3 | 60.0 |
| 11 | full | 25 | 378.1 | 58.2 |
| 12 | linear | 30 | 389.3 | 60.0 |
| 13 | linear | 30 | 389.3 | 60.0 |
| 14 | linear | 30 | 389.3 | 60.0 |
| 15 | full | 25 | 378.1 | 58.2 |
| 16 | linear | 30 | 389.3 | 60.0 |
| 17 | linear | 30 | 389.3 | 60.0 |
| 18 | linear | 30 | 389.3 | 60.0 |
| 19 | full | 25 | 378.1 | 58.2 |
| 20 | linear | 30 | 389.3 | 60.0 |
| 21 | linear | 30 | 389.3 | 60.0 |
| 22 | linear | 30 | 389.3 | 60.0 |
| 23 | full | 25 | 378.1 | 58.2 |
| 24 | linear | 30 | 389.3 | 60.0 |
| 25 | linear | 30 | 389.3 | 60.0 |
| 26 | linear | 30 | 389.3 | 60.0 |
| 27 | full | 25 | 378.1 | 58.2 |
| 28 | linear | 30 | 389.3 | 60.0 |
| 29 | linear | 30 | 389.3 | 60.0 |
| 30 | linear | 30 | 389.3 | 60.0 |
| 31 | full | 25 | 378.1 | 58.2 |
| 32 | linear | 30 | 389.3 | 60.0 |
| 33 | linear | 30 | 389.3 | 60.0 |
| 34 | linear | 30 | 389.3 | 60.0 |
| 35 | full | 25 | 378.1 | 58.2 |
| 36 | linear | 30 | 389.3 | 60.0 |
| 37 | linear | 30 | 389.3 | 60.0 |
| 38 | linear | 30 | 389.3 | 60.0 |
| 39 | full | 25 | 378.1 | 58.2 |
| 40 | linear | 30 | 389.3 | 60.0 |
| 41 | linear | 30 | 389.3 | 60.0 |
| 42 | linear | 30 | 389.3 | 60.0 |
| 43 | full | 25 | 378.1 | 58.2 |
| 44 | linear | 30 | 389.3 | 60.0 |
| 45 | linear | 30 | 389.3 | 60.0 |
| 46 | linear | 30 | 389.3 | 60.0 |
| 47 | full | 25 | 378.1 | 58.2 |
| 48 | linear | 30 | 389.3 | 60.0 |
| 49 | linear | 30 | 389.3 | 60.0 |
| 50 | linear | 30 | 389.3 | 60.0 |
| 51 | full | 25 | 378.1 | 58.2 |
| 52 | linear | 30 | 389.3 | 60.0 |
| 53 | linear | 30 | 389.3 | 60.0 |
| 54 | linear | 30 | 389.3 | 60.0 |
| 55 | full | 25 | 378.1 | 58.2 |
| 56 | linear | 30 | 389.3 | 60.0 |
| 57 | linear | 30 | 389.3 | 60.0 |
| 58 | linear | 30 | 389.3 | 60.0 |
| 59 | full | 25 | 378.1 | 58.2 |
| 60 | linear | 30 | 389.3 | 60.0 |
| 61 | linear | 30 | 389.3 | 60.0 |
| 62 | linear | 30 | 389.3 | 60.0 |
| 63 | full | 25 | 378.1 | 58.2 |

Tail (embed/lm_head/final norm): 2582.5 M logical params, 397.3 MB
