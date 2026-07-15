"""Shared primitives for bonsai-fold experiments."""

# Mandated generation convention for ALL Bonsai runs (CLAUDE.md / whitepaper
# App. B): this sampling AND thinking mode on. Use both helpers together so
# no harness can drift on half of the convention.
SAMPLING = {"temp": 0.7, "top_p": 0.95, "top_k": 20}
THINKING = True


def make_bonsai_sampler():
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(**SAMPLING)


def bonsai_prompt(tokenizer, messages):
    """Chat-template a message list with the mandated thinking mode."""
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=THINKING
    )
