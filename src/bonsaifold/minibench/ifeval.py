"""Rule-based verifiers for a subset of IFEval instruction types.

Reimplements the relevant checkers from google-research
instruction_following_eval (instructions.py) with the stdlib only, keyed by
the HF google/IFEval conventions: each item has an instruction_id_list (ids
like "keywords:existence") and a parallel kwargs list whose unused fields are
null-padded.

Deliberate deviations from the official code, all behavior-preserving on the
dataset's plain-word arguments:

- keyword / marker / section-splitter strings are re.escape()d before being
  interpolated into patterns (the official code inserts them raw).
- word count: official uses nltk RegexpTokenizer(r"\\w+");
  len(re.findall(r"\\w+", text)) is the exact stdlib equivalent.
- sentence count: official uses the nltk punkt tokenizer; here sentences are
  split on [.!?] followed by whitespace or end of text (simple heuristic, so
  abbreviations like "Dr." over-count relative to punkt).
"""
import json
import re


def _compare(count, n, relation):
    """Official _COMPARISON_RELATION semantics: "less than" | "at least"."""
    if relation == "less than":
        return count < n
    if relation == "at least":
        return count >= n
    raise ValueError(f"unknown relation: {relation!r}")


def _keywords_existence(text, keywords):
    # Case-insensitive substring search (official uses no word boundaries).
    return all(re.search(re.escape(k), text, flags=re.IGNORECASE) for k in keywords)


def _keywords_frequency(text, keyword, frequency, relation):
    count = len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))
    return _compare(count, frequency, relation)


def _forbidden_words(text, forbidden_words):
    # Whole-word match (\b boundaries), case-insensitive.
    return not any(
        re.search(r"\b" + re.escape(w) + r"\b", text, flags=re.IGNORECASE)
        for w in forbidden_words
    )


def _number_words(text, num_words, relation):
    return _compare(len(re.findall(r"\w+", text)), num_words, relation)


def _number_sentences(text, num_sentences, relation):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return _compare(sum(1 for p in parts if p.strip()), num_sentences, relation)


def _number_paragraphs(text, num_paragraphs):
    # Paragraphs are separated by the literal divider '***'. Blank first/last
    # segments are tolerated; a blank middle segment fails (official).
    parts = re.split(r"\s?\*\*\*\s?", text)
    count = len(parts)
    for i, p in enumerate(parts):
        if not p.strip():
            if i in (0, len(parts) - 1):
                count -= 1
            else:
                return False
    return count == num_paragraphs


def _number_bullet_lists(text, num_bullets):
    # Lines starting with '*' (but not '**') or '-'; exact count required.
    stars = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
    return len(stars) + len(dashes) == num_bullets


def _title(text):
    # A non-empty title wrapped in << >> anywhere in the response.
    titles = re.findall(r"<<[^\n]+>>", text)
    return any(t.lstrip("<").rstrip(">").strip() for t in titles)


def _json_format(text):
    text = text.strip()
    for fence in ("```json", "```Json", "```JSON", "```"):
        text = text.removeprefix(fence)
    text = text.removesuffix("```").strip()
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _number_highlighted_sections(text, num_highlights):
    # Non-empty *single* and **double** starred spans; at least num_highlights.
    count = 0
    for h in re.findall(r"\*[^\n\*]*\*", text):
        if h.strip("*").strip():
            count += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", text):
        if h.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


def _multiple_sections(text, section_spliter, num_sections):
    # section_spliter (dataset spelling) is e.g. "Section" or "SECTION",
    # marking headers like "Section 1"; at least num_sections required.
    pattern = r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?"
    return len(re.split(pattern, text)) - 1 >= num_sections


def _number_placeholders(text, num_placeholders):
    return len(re.findall(r"\[.*?\]", text)) >= num_placeholders


def _postscript(text, postscript_marker):
    text = text.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(postscript_marker.lower()) + r".*$"
    return bool(re.findall(pattern, text, flags=re.MULTILINE))


def _end_checker(text, end_phrase):
    return text.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def _quotation(text):
    text = text.strip()
    return len(text) > 1 and text[0] == '"' and text[-1] == '"'


def _english_lowercase(text):
    return text.islower()


def _english_capital(text):
    return text.isupper()


def _no_comma(text):
    return "," not in text


_CHECKERS = {
    "keywords:existence": _keywords_existence,
    "keywords:frequency": _keywords_frequency,
    "keywords:forbidden_words": _forbidden_words,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "detectable_format:number_bullet_lists": _number_bullet_lists,
    "detectable_format:title": _title,
    "detectable_format:json_format": _json_format,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "change_case:english_lowercase": _english_lowercase,
    "change_case:english_capital": _english_capital,
    "punctuation:no_comma": _no_comma,
}

SUPPORTED_TYPES = frozenset(_CHECKERS)


def verify(instruction_id, kwargs, response):
    """True if `response` satisfies one IFEval instruction.

    kwargs uses the HF google/IFEval field names; null-padded fields are
    dropped. Raises KeyError for an unsupported instruction_id.
    """
    checker = _CHECKERS[instruction_id]
    kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
    return bool(checker(response, **kwargs))


def item_supported(instruction_id_list):
    """True if every instruction id of an IFEval item is implemented here."""
    return all(i in SUPPORTED_TYPES for i in instruction_id_list)
