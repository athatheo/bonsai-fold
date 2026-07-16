"""Rule-first answer extraction and equivalence for thinking-mode LLM outputs.

Scoring core for the mini-bench (MATH-500, GSM8K, MMLU-Redux). Absolute
scores are compared across model variants, so everything here is:

- deterministic: pure string functions, no randomness, no external deps
  (stdlib ``re``/``fractions``/``decimal`` only, no sympy); never raises on
  model output (recursion depth and numeric size are capped);
- symmetric: ``math_equal(a, b) == math_equal(b, a)`` by construction (both
  sides go through the identical normalization pipeline and every comparison
  primitive is symmetric);
- conservative: when a normalization would be ambiguous we skip it and fall
  back to exact string comparison, preferring false negatives over false
  positives.

Normalizations applied by :func:`math_equal` (each documented at the site
where it happens in ``_normalize_once``; the pipeline is run to a fixpoint
so it is idempotent — required because tuple/interval elements are
re-normalized during recursion):

1. unicode minus/pi/degree mapped to ASCII/LaTeX
2. LaTeX no-ops dropped: ``\\left``/``\\right`` (token-aware, so
   ``\\leftarrow`` is untouched), spacing (``\\!``, ``\\,``, ``\\;``,
   ``\\:``, ``\\ ``, ``~``), ``\\dfrac``/``\\tfrac`` -> ``\\frac``
3. ``$`` (math delimiters / currency) and percent signs removed
4. degree markers removed (``^\\circ``, ``^{\\circ}``, U+00B0)
5. whitespace removed, EXCEPT between two digits ("(0, 1]" == "(0,1]" but
   "1 2" != "12")
6. units: TRAILING ``\\text{...}``/``\\mbox{...}`` blocks after other
   content are dropped ("5\\text{ cm}" == "5"); a standalone block is
   unwrapped ("\\text{even}" == "even"); mid-string blocks are kept
   verbatim so "2\\text{ or }3" never collapses to "23"
7. single-token ``\\sqrt``/``\\frac`` arguments braced per LaTeX semantics:
   ``\\sqrt2`` -> ``\\sqrt{2}``, ``\\frac12`` -> ``\\frac{1}{2}``,
   ``\\frac{\\sqrt{3}}2`` -> ``\\frac{\\sqrt{3}}{2}``
8. ONE trailing sentence period dropped ("42." == "42"); ellipses are
   preserved ("0.999..." != "0.999")
9. a whole-string ``\\frac{A}{B}`` becomes ``A/B``, but only when A and B
   are operator-free atoms — "\\frac{1+2}{3}" is NOT flattened to the
   grouping-ambiguous "1+2/3"
10. numeric comparison through exact ``Fraction(Decimal(...))`` arithmetic
    when both sides parse as a number or a single ratio of numbers
    ("2.5" == "5/2" == "\\frac{5}{2}"); thousands commas stripped only when
    the full token is a comma-grouped number ("1,234"); numeric tokens over
    64 chars or with exponents beyond 1e±40 are treated as non-numeric
    (falling back to exact string match) so degenerate outputs cannot stall
    the bench
11. tuples/intervals: identical delimiters required ("(0,1)" != "[0,1]"),
    then elementwise recursive comparison on top-level comma splits
12. everything else: exact string equality after the steps above (so
    "\\pi" == "2\\pi/2" is NOT accepted -- no symbolic algebra here)

Known conservative gaps and accepted collisions (intentional):
- "50%" != "0.5" (percent sign is dropped, the value is not rescaled);
- "\\sqrt{23}" != "\\sqrt23" (LaTeX reads the latter as sqrt(2)*3);
- no evaluation of pi, roots, or arithmetic expressions;
- trailing units are dropped from each side independently, so
  "5\\text{cm}" == "5\\text{km}" (standard MATH-style normalization --
  gold and prediction for one item share the expected unit in practice);
- $/%/degree signs are stripped independently, so "\\$5" == "5%"
  (same trade-off as above: within one item the symbol set matches).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

__all__ = [
    "strip_thinking",
    "extract_gsm8k",
    "gsm8k_gold",
    "extract_boxed",
    "math_equal",
    "extract_mmlu_choice",
]

_THINK_CLOSE = "</think>"
_UNICODE_MINUS = "−"

# Recursion cap for brace/tuple nesting. Real answers nest a handful of
# levels; degenerate repetition-loop outputs must score False, not crash.
_MAX_DEPTH = 32

# A number as models write it: optional sign and/or dollar sign, digits with
# optional thousands commas, optional decimal part, or a bare ".5". The
# negative-sign branch requires a non-word, non-digit left context so range
# hyphens are not read as signs ("pages 10-25" yields 25, not -25).
_NUMBER_RE = re.compile(
    r"""
    (?:
        (?<![\w.])(?:\$-|-\$|-)       # sign (with optional $), not glued to a word/number
      | \$                            # bare currency prefix
    )?
    (?:
        (?:\d{1,3}(?:,\d{3})+|\d+)    # integer part, optionally comma-grouped
        (?:\.\d+)?                    # optional decimal part
      | \.\d+                         # or a bare decimal like .5
    )
    """,
    re.VERBOSE,
)

# A plain (non-comma) numeric literal for exact parsing. Anchored via
# fullmatch, so "nan"/"inf" and stray text can never reach Decimal().
_PLAIN_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_EXPONENT_RE = re.compile(r"[eE]([-+]?\d+)$")

# A comma-grouped integer/decimal ("1,234", "-12,345.6"). Commas are only
# stripped when the *entire* token matches, so tuple separators survive.
_GROUPED_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")

# "the answer is X" / "answer is: X" / "answer: X" marker, shared by the
# free-form and MMLU extractors so the accepted phrasing cannot drift
# between them. "is" requires a word boundary so "isn't"/"island" never
# match. The payload is captured separately (from the LAST marker onward)
# so earlier matches cannot swallow later ones.
_ANSWER_MARKER = r"\banswer\s*(?:is\b\s*:?|:)\s*"
_ANSWER_IS_RE = re.compile(_ANSWER_MARKER, re.IGNORECASE)

# Gold marker capture confined to the marker's own line: \s would otherwise
# cross the newline after a bare "####" and capture a later line (including
# a subsequent marker), defeating last-marker-wins.
_GSM8K_GOLD_RE = re.compile(r"####[ \t]*([^\n]+)")
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
# First sentence of a captured line: stop at a period followed by whitespace
# or EOL, so decimals like "2.5" survive.
_SENTENCE_RE = re.compile(r"(.+?)(?:\.\s|\.$|$)")


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Final match of ``pattern`` in ``text``, or None.

    The module-wide extraction policy: thinking-mode models routinely state
    wrong candidates before committing, so the LAST occurrence wins.
    """
    last = None
    for last in pattern.finditer(text):
        pass
    return last


def strip_thinking(text: str) -> str:
    """Return the content after the LAST '</think>' tag.

    If no tag is present, the whole text is returned unchanged (some models
    omit the opening tag, or thinking mode may have been disabled).
    """
    idx = text.rfind(_THINK_CLOSE)
    if idx == -1:
        return text
    return text[idx + len(_THINK_CLOSE):]


def _clean_number(token: str) -> str:
    """Strip currency/commas from a number token; keep a LEADING sign only.

    Interior hyphens are preserved ("3-4" stays "3-4", it does not become
    "-34"), so range-like gold fields fail comparison instead of silently
    turning into a different negative number.
    """
    token = token.replace("$", "").replace(",", "").strip()
    negative = token.startswith("-")
    return ("-" if negative else "") + token.lstrip("-")


def extract_gsm8k(answer_text: str) -> str | None:
    """Model-side GSM8K extraction: the LAST number in the text.

    Accepts dollar signs, thousands commas, decimals, and negatives;
    returns the number with '$' and ',' stripped (e.g. "$1,234.50" ->
    "1234.50"). Returns None if the text contains no number. Compare
    against :func:`gsm8k_gold` output with :func:`math_equal`.
    """
    m = _last_match(_NUMBER_RE, answer_text.replace(_UNICODE_MINUS, "-"))
    if m is None:
        return None
    return _clean_number(m.group(0))


def gsm8k_gold(gold_field: str) -> str:
    """Dataset-side GSM8K gold: the number after the LAST '#### ' marker.

    Strips '$', thousands commas, and surrounding whitespace. Raises
    ValueError if the field has no '####' marker, so malformed dataset rows
    surface instead of silently scoring as wrong.
    """
    m = _last_match(_GSM8K_GOLD_RE, gold_field)
    if m is None:
        raise ValueError(f"no '####' marker in GSM8K gold field: {gold_field!r}")
    return _clean_number(m.group(1).strip())


def _consume_group(text: str, open_idx: int) -> tuple[str, int] | None:
    """Content and end index of the brace group whose '{' is at ``open_idx``.

    Returns ``(content, index_past_closing_brace)``. Handles nesting; a
    backslash escapes the following character (so literal ``\\{``/``\\}`` do
    not affect the depth count). Returns None when there is no group at
    ``open_idx`` or the group never closes.
    """
    if open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
        i += 1
    return None


def _trim_top_level_clause(payload: str) -> str:
    """Cut an 'answer is X' payload at the first top-level ', ' — a comma
    followed by a space at bracket depth 0 marks a prose continuation
    ("7, so we are done" -> "7"). Commas inside tuples/intervals and
    thousands separators (no following space) are preserved."""
    depth = 0
    for i, c in enumerate(payload):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0 and payload[i + 1 : i + 2] == " ":
            return payload[:i]
    return payload


def _strip_trailing_words(candidate: str) -> str:
    """Drop trailing purely-alphabetic words (unit/noun tails such as
    "18 eggs" -> "18") while the remainder still contains a digit. Answers
    with no digits ("x plus y") and mixed tails ("2 or 3") are untouched."""
    words = candidate.split(" ")
    while (
        len(words) > 1
        and words[-1].isalpha()
        and len(words[-1]) >= 2
        and any(ch.isdigit() for ch in " ".join(words[:-1]))
    ):
        words.pop()
    return " ".join(words)


def _last_answer_is(text: str) -> str | None:
    """Payload of the LAST 'answer is X' / 'answer: X', trimmed to the
    sentence, the top-level clause, and stripped of markdown emphasis and
    trailing unit words."""
    marker = _last_match(_ANSWER_IS_RE, text)
    if marker is None:
        return None
    rest = text[marker.end():].split("\n", 1)[0]
    m = _SENTENCE_RE.match(rest)
    candidate = _trim_top_level_clause(m.group(1) if m else rest)
    candidate = candidate.strip().strip("*").strip()
    candidate = _strip_trailing_words(candidate)
    return candidate or None


def extract_boxed(answer_text: str) -> str | None:
    """MATH-style extraction: content of the LAST ``\\boxed{...}``.

    Brace-balanced, so nested groups like ``\\boxed{\\frac{1}{2}}`` come out
    whole. Unclosed/empty boxes are skipped in favor of earlier ones. Falls
    back to the last 'answer is X' pattern, then the last number in the
    text, else None. NOTE: because of these fallbacks, this function is
    None only when :func:`extract_gsm8k` is also None.
    """
    for m in reversed(list(_BOXED_OPEN_RE.finditer(answer_text))):
        group = _consume_group(answer_text, m.end() - 1)
        if group is not None and group[0].strip():
            return group[0].strip()
    ans = _last_answer_is(answer_text)
    if ans is not None:
        return ans
    return extract_gsm8k(answer_text)


# ---------------------------------------------------------------------------
# math_equal
# ---------------------------------------------------------------------------

# Token/character canonicalization applied first by _normalize_once. Every
# entry is value-preserving, and no entry's LHS is a prefix of a longer
# LaTeX command (commands are backslash+letters, so "\\," etc. are complete
# tokens; \\left/\\right need the token-aware regex below instead). ORDER
# MATTERS within the table: escaped forms ("\\$", "\\%") before bare ones.
_CANON_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (_UNICODE_MINUS, "-"),  # unicode minus
    ("π", "\\pi"),          # pi symbol == \pi (textual, not numeric)
    ("°", ""),              # degree sign
    ("\\!", ""),            # LaTeX spacing no-ops
    ("\\,", ""),
    ("\\;", ""),
    ("\\:", ""),
    ("\\ ", ""),
    ("~", ""),
    ("\\dfrac", "\\frac"),  # display/text-style fractions == \frac
    ("\\tfrac", "\\frac"),
    ("\\$", ""),            # math delimiters / currency
    ("$", ""),
    ("\\%", ""),            # percent sign dropped; the value is NOT
    ("%", ""),              # rescaled, so "50%" == "50" but != "0.5"
)

# \left / \right sizing no-ops. Token-aware: must not be followed by a
# letter, so \leftarrow / \rightarrow are left intact.
_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right)(?![a-zA-Z])")

# Degree markers: 45^\circ == 45^{\circ} == 45.
_DEGREE_RE = re.compile(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ)")

# Whitespace BETWEEN TWO DIGITS is preserved (as a single space) so "1 2"
# never collapses into "12"; all other whitespace is removed.
_DIGIT_GAP_RE = re.compile(r"(?<=\d)\s+(?=\d)")
_WHITESPACE_RE = re.compile(r"\s+")
_DIGIT_GAP_SENTINEL = "\x00"

_TEXT_UNWRAP_RE = re.compile(r"\\(?:text|mbox)\{([^{}]*)\}")
_TRAILING_TEXT_RE = re.compile(r"\\(?:text|mbox)\{[^{}]*\}\Z")


def _strip_text_blocks(s: str) -> str:
    """Unit handling for \\text{...}/\\mbox{...} (whitespace already removed).

    - standalone block: unwrapped ("\\text{even}" -> "even");
    - TRAILING blocks after other content: dropped ("5\\text{cm}" -> "5");
    - mid-string blocks: kept verbatim, so "2\\text{or}3" never becomes
      "23" (they fall through to exact string comparison, and tuple
      elements like "(\\text{a},\\text{b})" still unwrap per element).
    """
    m = _TEXT_UNWRAP_RE.fullmatch(s)
    if m:
        return m.group(1)
    while True:
        t = _TRAILING_TEXT_RE.search(s)
        if t is None or t.start() == 0:
            return s
        s = s[: t.start()]


def _brace_command_args(s: str, command: str, nargs: int, depth: int = 0) -> str:
    """Rewrite every ``\\<command>`` occurrence so its arguments are braced.

    Per LaTeX single-token argument semantics: ``\\sqrt2`` -> ``\\sqrt{2}``,
    ``\\frac12`` -> ``\\frac{1}{2}``, ``\\frac{\\sqrt{3}}2`` ->
    ``\\frac{\\sqrt{3}}{2}``. An argument is either a balanced ``{...}``
    group (processed recursively, so same-command nesting is normalized too)
    or a single ``[0-9a-zA-Z]`` token. Occurrences whose arguments fit
    neither shape (e.g. ``\\sqrt[3]{8}``, or a truncated ``\\frac`` at end
    of string) are left verbatim -- conservative fall-through to exact
    string comparison. Nesting beyond _MAX_DEPTH is left verbatim too
    (never raise on degenerate output).
    """
    if depth >= _MAX_DEPTH:
        return s
    token = "\\" + command
    out: list[str] = []
    i = 0
    while True:
        j = s.find(token, i)
        if j == -1:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        pos = j + len(token)
        args: list[str] | None = []
        for _ in range(nargs):
            group = _consume_group(s, pos)
            if group is not None:
                args.append(_brace_command_args(group[0], command, nargs, depth + 1))
                pos = group[1]
            elif pos < len(s) and s[pos].isascii() and s[pos].isalnum():
                args.append(s[pos])
                pos += 1
            else:
                args = None
                break
        if args is None:
            out.append(token)
            i = j + len(token)
        else:
            out.append(token + "".join("{" + a + "}" for a in args))
            i = pos


def _atomic_operand(t: str) -> bool:
    """True when ``t`` (an optionally-negated fraction operand) contains no
    binary operator, so flattening it next to a '/' cannot lose grouping."""
    body = t[1:] if t.startswith("-") else t
    return bool(body) and not any(op in body for op in "+-*/")


def _frac_to_slash(s: str) -> str:
    """Whole-string ``\\frac{A}{B}`` (optionally negated) -> ``A/B``.

    Only fires when the fraction spans the entire string AND both operands
    are operator-free atoms: "\\frac{1}{2}x" is left alone (trailing
    content) and so is "\\frac{1+2}{3}" (flattening to "1+2/3" would lose
    the grouping and could equate different values).
    """
    sign = ""
    body = s
    if body.startswith("-"):
        sign, body = "-", body[1:]
    prefix = "\\frac"
    if not body.startswith(prefix + "{"):
        return s
    num = _consume_group(body, len(prefix))
    if num is None or not num[0]:
        return s
    den = _consume_group(body, num[1])
    if den is None or not den[0]:
        return s
    if den[1] != len(body):  # trailing content -> not a pure fraction
        return s
    if not (_atomic_operand(num[0]) and _atomic_operand(den[0])):
        return s
    return f"{sign}{num[0]}/{den[0]}"


def _normalize_once(s: str) -> str:
    """One pass of the normalization pipeline (see module docstring)."""
    s = s.strip()
    # 1-3. canonical characters, LaTeX no-ops, $/% removal (table-driven)
    for old, new in _CANON_REPLACEMENTS:
        s = s.replace(old, new)
    s = _LEFT_RIGHT_RE.sub("", s)
    # 4. degree markers
    s = _DEGREE_RE.sub("", s)
    # 5. remove whitespace, preserving a single gap between adjacent digits
    s = _DIGIT_GAP_RE.sub(_DIGIT_GAP_SENTINEL, s)
    s = _WHITESPACE_RE.sub("", s)
    s = s.replace(_DIGIT_GAP_SENTINEL, " ")
    # 6. units via \text{...}/\mbox{...} (after whitespace removal, so
    #    "\text {cm}" and "\text{cm}" behave identically)
    s = _strip_text_blocks(s)
    # 7. brace single-token \sqrt/\frac arguments per LaTeX semantics
    s = _brace_command_args(s, "sqrt", 1)
    s = _brace_command_args(s, "frac", 2)
    # 8. ONE trailing sentence period ("42." == "42"); a trailing ellipsis
    #    is meaningful ("0.999..." is not "0.999") and is kept
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1]
    # 9. whole-string \frac{A}{B} -> A/B (atomic operands only)
    s = _frac_to_slash(s)
    return s


def _normalize(s: str) -> str:
    """Run the pipeline to a fixpoint so normalization is idempotent (tuple
    elements are re-normalized during recursion and must not drift). In
    practice one or two passes converge; the bound keeps it total."""
    for _ in range(4):
        out = _normalize_once(s)
        if out == s:
            return out
        s = out
    return s


def _plain_to_fraction(s: str) -> Fraction | None:
    """Exact Fraction for a plain numeric literal, else None.

    Decimal handles leading zeros ("007"), bare-dot decimals (".5"), and
    scientific notation exactly; Fraction(Decimal) is exact, so 2.5 == 5/2
    holds without any float rounding. Tokens longer than 64 chars or with
    exponents beyond 1e+-40 are rejected: Fraction(Decimal("1e10000000"))
    would materialize the full integer and stall the bench, and no
    benchmark answer needs that range.
    """
    if len(s) > 64 or not _PLAIN_NUMBER_RE.fullmatch(s):
        return None
    exp = _EXPONENT_RE.search(s)
    if exp is not None and abs(int(exp.group(1))) > 40:
        return None
    try:
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError):
        return None


def _to_fraction(s: str) -> Fraction | None:
    """Exact numeric value of a normalized token: plain number, comma-grouped
    number, or a single ratio 'a/b' of plain numbers. None if non-numeric."""
    if _GROUPED_NUMBER_RE.fullmatch(s):
        s = s.replace(",", "")
    if "/" in s:
        left, _, right = s.partition("/")
        if "/" in right:
            return None
        lf = _plain_to_fraction(left)
        rf = _plain_to_fraction(right)
        if lf is None or rf is None or rf == 0:
            return None
        return lf / rf
    return _plain_to_fraction(s)


def _split_top_level(s: str) -> list[str] | None:
    """Split on commas at bracket depth 0. None if brackets are unbalanced."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for c in s:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return None
        if c == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    if depth != 0:
        return None
    parts.append("".join(buf))
    return parts


def _equal_normalized(p: str, g: str, depth: int = 0) -> bool:
    if not p or not g:
        # An empty side never matches anything, including another empty side
        # (an empty prediction is a failed extraction, not a correct answer).
        return False
    if p == g:
        return True
    if depth >= _MAX_DEPTH:
        # Degenerate nesting: score False rather than recurse without bound.
        return False
    pf = _to_fraction(p)
    gf = _to_fraction(g)
    if pf is not None and gf is not None:
        return pf == gf
    # Tuples / intervals: delimiters must match exactly -- "(0,1)" is an open
    # interval and "[0,1]" a closed one, so mixed brackets never compare equal.
    if (
        len(p) >= 2
        and len(g) >= 2
        and p[0] == g[0]
        and p[-1] == g[-1]
        and p[0] in "(["
        and p[-1] in ")]"
    ):
        pe = _split_top_level(p[1:-1])
        ge = _split_top_level(g[1:-1])
        if pe is not None and ge is not None and len(pe) == len(ge):
            # Re-normalize elements: whole-string rules (e.g. \frac -> a/b,
            # standalone \text unwrap) now apply per element. _normalize is
            # idempotent (fixpoint), so re-application cannot drift.
            return all(
                _equal_normalized(_normalize(a), _normalize(b), depth + 1)
                for a, b in zip(pe, ge)
            )
    return False


def math_equal(pred: str, gold: str) -> bool:
    """Conservative, deterministic, symmetric answer equivalence.

    Both sides run through the same normalization pipeline, then compare by
    exact string match, exact rational arithmetic (when both sides are
    numeric), or elementwise tuple/interval recursion. No symbolic algebra:
    unparseable forms must match textually after normalization. Never
    raises on any string input.
    """
    if not isinstance(pred, str) or not isinstance(gold, str):
        return False
    return _equal_normalized(_normalize(pred), _normalize(gold))


# ---------------------------------------------------------------------------
# MMLU choice extraction
# ---------------------------------------------------------------------------

# Tier 1 (strongest): "the answer is (C)" / "Answer: C" / "answer is **b**".
# The trailing guard excludes letters AND digits so content words like
# "B12" or "C4" are not read as choices.
_MMLU_ANSWER_RE = re.compile(
    _ANSWER_MARKER + r"\**\(?\s*([A-D])\s*\)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Tier 2: a parenthesized letter, "(C)". The left guard excludes identifier
# characters so function/probability notation like "P(B)" or "f(a)" does
# not register as a choice.
_MMLU_PAREN_RE = re.compile(r"(?<![A-Za-z0-9_])\(([A-Da-d])\)")
# Tier 3 (weakest): a bare letter ending the text ("...\n\nC." / "... C").
# Uppercase only -- a bare trailing lowercase letter is too ambiguous.
_MMLU_TRAILING_RE = re.compile(r"(?:^|[\s*(])([A-D])[.)\]*]*\s*$")


def extract_mmlu_choice(answer_text: str) -> str | None:
    """Letter A-D from a model response, or None.

    Patterns are tried strongest-first ('answer is X' > '(X)' > bare
    trailing letter); within a tier the LAST occurrence wins, since
    thinking-mode models routinely mention wrong options before committing.
    """
    for pattern in (_MMLU_ANSWER_RE, _MMLU_PAREN_RE):
        m = _last_match(pattern, answer_text)
        if m is not None:
            return m.group(1).upper()
    m = _MMLU_TRAILING_RE.search(answer_text.rstrip())
    if m:
        return m.group(1)
    return None
