"""Tests for bonsaifold.minibench.answers — the mini-bench scoring core."""

import pytest

from bonsaifold.minibench.answers import (
    extract_boxed,
    extract_gsm8k,
    extract_mmlu_choice,
    gsm8k_gold,
    math_equal,
    strip_thinking,
)

# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------


class TestStripThinking:
    def test_no_tag_returns_whole_text(self):
        assert strip_thinking("just an answer") == "just an answer"

    def test_single_tag(self):
        assert strip_thinking("<think>hmm 12</think>The answer is 5.") == "The answer is 5."

    def test_multiple_tags_uses_last(self):
        text = "<think>a</think>draft</think>final"
        assert strip_thinking(text) == "final"

    def test_missing_open_tag(self):
        assert strip_thinking("secretly thinking</think>42") == "42"

    def test_text_ending_with_tag(self):
        assert strip_thinking("<think>all thinking</think>") == ""

    def test_multiline_content_after_tag(self):
        text = "<think>x</think>\nline1\nline2"
        assert strip_thinking(text) == "\nline1\nline2"

    def test_empty_string(self):
        assert strip_thinking("") == ""


# ---------------------------------------------------------------------------
# extract_gsm8k (model side)
# ---------------------------------------------------------------------------


class TestExtractGsm8k:
    def test_last_number_wins(self):
        assert extract_gsm8k("He had 3 apples, ate 1, so 2 remain.") == "2"

    def test_dollar_and_commas(self):
        assert extract_gsm8k("The total is $1,234.50") == "1234.50"

    def test_dollar_after_minus(self):
        assert extract_gsm8k("net change of -$40") == "-40"

    def test_negative(self):
        assert extract_gsm8k("the temperature drops to -3") == "-3"

    def test_decimal(self):
        assert extract_gsm8k("which gives 2.5 hours") == "2.5"

    def test_bare_dot_decimal(self):
        assert extract_gsm8k("a probability of .5") == ".5"

    def test_trailing_sentence_period_not_included(self):
        assert extract_gsm8k("So the answer is 42.") == "42"

    def test_comma_grouped_not_confused_with_list(self):
        # "3, 4" is a list, not the number 34
        assert extract_gsm8k("we have 3, 4 and then 72 eggs") == "72"

    def test_no_number_returns_none(self):
        assert extract_gsm8k("no digits here at all") is None

    def test_empty_returns_none(self):
        assert extract_gsm8k("") is None


class TestGsm8kGold:
    def test_plain(self):
        assert gsm8k_gold("She has 72 eggs left.\n#### 72") == "72"

    def test_commas_stripped(self):
        assert gsm8k_gold("blah\n#### 1,234") == "1234"

    def test_dollar_stripped(self):
        assert gsm8k_gold("#### $500") == "500"

    def test_negative(self):
        assert gsm8k_gold("#### -5") == "-5"

    def test_last_marker_wins(self):
        assert gsm8k_gold("#### 1\nmore work\n#### 2") == "2"

    def test_missing_marker_raises(self):
        with pytest.raises(ValueError):
            gsm8k_gold("no marker here")


# ---------------------------------------------------------------------------
# extract_boxed
# ---------------------------------------------------------------------------


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"Thus \boxed{42}.") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"So \boxed{\frac{1}{2}} is it.") == r"\frac{1}{2}"

    def test_deeply_nested(self):
        assert extract_boxed(r"\boxed{\sqrt{a_{1}^{2}+b}}") == r"\sqrt{a_{1}^{2}+b}"

    def test_last_boxed_wins(self):
        assert extract_boxed(r"maybe \boxed{1}... no, \boxed{2}") == "2"

    def test_space_before_brace(self):
        assert extract_boxed(r"\boxed {7}") == "7"

    def test_unclosed_box_falls_back_to_earlier_box(self):
        assert extract_boxed(r"\boxed{3} then \boxed{\frac{1}{2}") == "3"

    def test_empty_box_skipped(self):
        assert extract_boxed(r"\boxed{5} oops \boxed{}") == "5"

    def test_fallback_answer_is(self):
        assert extract_boxed("The answer is 7.") == "7"

    def test_fallback_answer_is_latex(self):
        assert extract_boxed(r"The answer is $\frac{1}{2}$.") == r"$\frac{1}{2}$"

    def test_fallback_answer_is_takes_last(self):
        assert extract_boxed("The answer is 3. Wait, the answer is 4.") == "4"

    def test_fallback_last_number(self):
        assert extract_boxed("we compute 12 then 15") == "15"

    def test_nothing_returns_none(self):
        assert extract_boxed("no conclusion was reached") is None


# ---------------------------------------------------------------------------
# math_equal
# ---------------------------------------------------------------------------

EQUAL_PAIRS = [
    # exact / whitespace / delimiters
    ("42", "42"),
    (" 42 ", "42"),
    ("$16$", "16"),
    (r"\left(1,2\right)", "(1,2)"),
    # fractions vs decimals (exact rational arithmetic)
    ("2.5", "5/2"),
    (r"\frac{5}{2}", "2.5"),
    (r"\frac{1}{2}", "1/2"),
    (r"\dfrac{1}{2}", r"\frac{1}{2}"),
    (r"-\frac{1}{2}", "-0.5"),
    (r"\frac12", "0.5"),
    (r"\frac{1}2", "1/2"),
    (r"\frac1{2}", "0.5"),
    (r"\frac{\sqrt{3}}2", r"\frac{\sqrt{3}}{2}"),
    ("1/4", "0.25"),
    # numeric formatting
    ("0.5", ".5"),
    ("007", "7"),
    ("42.", "42"),
    ("1,234", "1234"),
    ("-3", "-3.0"),
    ("100", "100.00"),
    # percent (sign stripped, value NOT rescaled)
    (r"50\%", "50"),
    ("50%", "50"),
    # degrees
    (r"45^\circ", "45"),
    (r"45^{\circ}", "45"),
    ("45°", "45"),
    # pi (textual)
    (r"\pi", "π"),
    (r"2\pi", "2π"),
    (r"\frac{\pi}{2}", "π/2"),
    # sqrt forms
    (r"\sqrt{2}", r"\sqrt2"),
    (r"3\sqrt{2}", r"3\sqrt2"),
    (r"\frac{\sqrt{3}}{2}", r"\sqrt{3}/2"),
    # units
    (r"5\text{ cm}", "5"),
    (r"\text{even}", "even"),
    # tuples and intervals (elementwise)
    ("(1, 2)", "(1,2)"),
    ("[0, 1)", "[0,1)"),
    ("(1/2, 3)", r"(\frac{1}{2}, 3)"),
    ("(0.5, -2)", "(1/2, -2.0)"),
    ("[2, 2.5]", "[2, 5/2]"),
]

UNEQUAL_PAIRS = [
    ("1/2", "1/3"),
    ("2.5", "2.05"),
    (r"\frac{1}{2}", r"\frac{2}{1}"),
    ("-1/2", "1/2"),
    ("-3", "3"),
    (r"\sqrt{2}", "2"),
    (r"\sqrt{2}", r"\sqrt{3}"),
    (r"\pi", "3.14159"),
    (r"2\pi", r"\pi"),
    # interval bracket types are meaningful
    ("(0,1)", "[0,1]"),
    ("[0,1)", "(0,1]"),
    # tuple order and arity matter
    ("(1,2)", "(2,1)"),
    ("(1,2)", "(1,2,3)"),
    # percent is not rescaled
    ("50%", "0.5"),
    # no symbolic algebra
    ("x+1", "x+2"),
    ("2+3", "5"),
    ("", "5"),
    ("", ""),
]


class TestMathEqual:
    @pytest.mark.parametrize("a,b", EQUAL_PAIRS)
    def test_equal(self, a, b):
        assert math_equal(a, b), f"expected {a!r} == {b!r}"

    @pytest.mark.parametrize("a,b", UNEQUAL_PAIRS)
    def test_unequal(self, a, b):
        assert not math_equal(a, b), f"expected {a!r} != {b!r}"

    @pytest.mark.parametrize("a,b", EQUAL_PAIRS + UNEQUAL_PAIRS)
    def test_symmetry(self, a, b):
        assert math_equal(a, b) == math_equal(b, a)

    @pytest.mark.parametrize("a,b", EQUAL_PAIRS + UNEQUAL_PAIRS)
    def test_deterministic(self, a, b):
        assert math_equal(a, b) == math_equal(a, b)

    def test_none_inputs_are_false(self):
        assert not math_equal(None, "5")
        assert not math_equal("5", None)

    def test_division_by_zero_is_not_numeric(self):
        # falls back to string comparison, never raises
        assert math_equal("1/0", "1/0")
        assert not math_equal("1/0", "2/0")

    def test_no_zero_padding_false_positive(self):
        assert not math_equal("10", "1")

    def test_frac_with_trailing_factor_not_collapsed(self):
        # \frac{1}{2}x is not the scalar 1/2
        assert not math_equal(r"\frac{1}{2}x", "1/2")


# ---------------------------------------------------------------------------
# extract_mmlu_choice
# ---------------------------------------------------------------------------


class TestExtractMmluChoice:
    def test_answer_is_paren(self):
        assert extract_mmlu_choice("The answer is (C)") == "C"

    def test_answer_is_bare(self):
        assert extract_mmlu_choice("So the answer is B.") == "B"

    def test_answer_colon(self):
        assert extract_mmlu_choice("Answer: D") == "D"

    def test_answer_colon_paren(self):
        assert extract_mmlu_choice("Answer: (A)") == "A"

    def test_lowercase_normalized(self):
        assert extract_mmlu_choice("the answer is c") == "C"

    def test_markdown_bold(self):
        assert extract_mmlu_choice("**Answer:** B") == "B"

    def test_strong_pattern_beats_later_paren(self):
        text = "The answer is A because (B) and (C) are wrong."
        assert extract_mmlu_choice(text) == "A"

    def test_last_strong_occurrence_wins(self):
        text = "The answer is A. Wait — actually the answer is D."
        assert extract_mmlu_choice(text) == "D"

    def test_paren_only(self):
        assert extract_mmlu_choice("I would go with (B) here") == "B"

    def test_last_paren_wins(self):
        assert extract_mmlu_choice("not (A), rather (C)") == "C"

    def test_bare_trailing_letter(self):
        assert extract_mmlu_choice("Weighing the options...\n\nC") == "C"

    def test_bare_trailing_letter_with_period(self):
        assert extract_mmlu_choice("the answer must be\n\nC.") == "C"

    def test_word_letters_not_matched(self):
        assert extract_mmlu_choice("the answer is Considered unknown") is None

    def test_letter_out_of_range(self):
        assert extract_mmlu_choice("The answer is (E)") is None

    def test_no_letter_returns_none(self):
        assert extract_mmlu_choice("I really cannot tell.") is None

    def test_empty_returns_none(self):
        assert extract_mmlu_choice("") is None

    def test_full_pipeline_with_thinking(self):
        raw = "<think>options... (A)? no. (B)? maybe.</think>The answer is (D)."
        assert extract_mmlu_choice(strip_thinking(raw)) == "D"
