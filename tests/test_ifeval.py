import pytest

from bonsaifold.minibench.ifeval import SUPPORTED_TYPES, item_supported, verify


def test_keywords_existence():
    kw = {"keywords": ["forest", "river"]}
    assert verify("keywords:existence", kw, "The Forest meets the river.")  # case-insensitive
    assert not verify("keywords:existence", kw, "The forest is dry.")


def test_keywords_frequency():
    text = "cat cat cat"
    assert verify("keywords:frequency", {"keyword": "cat", "frequency": 3, "relation": "at least"}, text)
    assert not verify("keywords:frequency", {"keyword": "cat", "frequency": 4, "relation": "at least"}, text)
    assert verify("keywords:frequency", {"keyword": "cat", "frequency": 4, "relation": "less than"}, text)
    assert not verify("keywords:frequency", {"keyword": "cat", "frequency": 3, "relation": "less than"}, text)


def test_keywords_forbidden_words():
    kw = {"forbidden_words": ["dog"]}
    assert verify("keywords:forbidden_words", kw, "A dogged cat.")  # whole-word only
    assert not verify("keywords:forbidden_words", kw, "A Dog barks.")


def test_number_words():
    text = "one two three four five"
    assert verify("length_constraints:number_words", {"num_words": 5, "relation": "at least"}, text)
    assert not verify("length_constraints:number_words", {"num_words": 6, "relation": "at least"}, text)
    assert verify("length_constraints:number_words", {"num_words": 6, "relation": "less than"}, text)
    assert not verify("length_constraints:number_words", {"num_words": 5, "relation": "less than"}, text)


def test_number_sentences():
    text = "First one. Second one! Third one?"
    assert verify("length_constraints:number_sentences", {"num_sentences": 3, "relation": "at least"}, text)
    assert not verify("length_constraints:number_sentences", {"num_sentences": 4, "relation": "at least"}, text)
    assert verify("length_constraints:number_sentences", {"num_sentences": 2, "relation": "less than"}, "One sentence.")
    assert not verify("length_constraints:number_sentences", {"num_sentences": 3, "relation": "less than"}, text)


def test_number_paragraphs():
    text = "Para one.\n***\nPara two.\n***\nPara three."
    assert verify("length_constraints:number_paragraphs", {"num_paragraphs": 3}, text)
    assert not verify("length_constraints:number_paragraphs", {"num_paragraphs": 2}, text)
    # blank trailing segment tolerated, blank middle segment fails
    assert verify("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "One.\n***\nTwo.\n***\n")
    assert not verify("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "One.\n***\n\n***\nTwo.")


def test_number_bullet_lists():
    text = "Intro:\n* first\n* second\n- third"
    assert verify("detectable_format:number_bullet_lists", {"num_bullets": 3}, text)
    assert not verify("detectable_format:number_bullet_lists", {"num_bullets": 2}, text)
    # '**bold**' lines are not bullets
    assert verify("detectable_format:number_bullet_lists", {"num_bullets": 1}, "**bold**\n* one")


def test_title():
    assert verify("detectable_format:title", {}, "<<My Great Title>>\nBody text.")
    assert not verify("detectable_format:title", {}, "No title here.")
    assert not verify("detectable_format:title", {}, "<< >>")  # empty title


def test_json_format():
    assert verify("detectable_format:json_format", {}, '{"a": [1, 2], "b": null}')
    assert verify("detectable_format:json_format", {}, '```json\n{"a": 1}\n```')
    assert not verify("detectable_format:json_format", {}, "not json at all")


def test_number_highlighted_sections():
    text = "This has *one* and **two** highlights."
    assert verify("detectable_format:number_highlighted_sections", {"num_highlights": 2}, text)
    assert not verify("detectable_format:number_highlighted_sections", {"num_highlights": 3}, text)
    assert not verify("detectable_format:number_highlighted_sections", {"num_highlights": 1}, "empty ** only")


def test_multiple_sections():
    text = "Section 1\nintro\nSection 2\nmore\nSection 3\nend"
    kw = {"section_spliter": "Section", "num_sections": 3}
    assert verify("detectable_format:multiple_sections", kw, text)
    assert not verify("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 4}, text)
    assert not verify("detectable_format:multiple_sections", {"section_spliter": "SECTION", "num_sections": 3}, text)


def test_number_placeholders():
    text = "Dear [name], meet me at [address]."
    assert verify("detectable_content:number_placeholders", {"num_placeholders": 2}, text)
    assert not verify("detectable_content:number_placeholders", {"num_placeholders": 3}, text)


def test_postscript():
    assert verify("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body text.\n\nP.S. Bring snacks.")
    assert not verify("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body text only.")
    assert verify("detectable_content:postscript", {"postscript_marker": "P.P.S"}, "Text.\nP.P.S. Also this.")


def test_end_checker():
    kw = {"end_phrase": "That is all."}
    assert verify("startend:end_checker", kw, "Blah blah. that is all.")  # case-insensitive
    assert not verify("startend:end_checker", kw, "That is all. Blah blah.")


def test_quotation():
    assert verify("startend:quotation", {}, '  "Fully wrapped."  ')
    assert not verify("startend:quotation", {}, 'She said "hi" to me.')


def test_english_lowercase():
    assert verify("change_case:english_lowercase", {}, "all lower case here.")
    assert not verify("change_case:english_lowercase", {}, "Not all Lower.")


def test_english_capital():
    assert verify("change_case:english_capital", {}, "ALL CAPS HERE!")
    assert not verify("change_case:english_capital", {}, "Not ALL caps")


def test_no_comma():
    assert verify("punctuation:no_comma", {}, "No commas here.")
    assert not verify("punctuation:no_comma", {}, "One, comma.")


def test_none_padded_kwargs_ignored():
    # HF dataset pads unused kwargs fields with null
    assert verify("detectable_format:title", {"num_highlights": None, "relation": None}, "<<T>>")


def test_verify_unsupported_raises():
    with pytest.raises(KeyError):
        verify("keywords:letter_frequency", {}, "anything")


def test_item_supported():
    assert item_supported(["keywords:existence", "punctuation:no_comma"])
    assert not item_supported(["keywords:existence", "keywords:letter_frequency"])
    assert item_supported([])
    assert len(SUPPORTED_TYPES) == 18
