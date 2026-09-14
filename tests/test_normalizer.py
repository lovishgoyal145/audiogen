"""Unit tests for Indic text normalization module (Hindi and Punjabi)."""

import pytest
from audiogen.normalizer import (
    IndicNormalizer,
    expand_digits,
    normalize_text,
    number_to_words,
    strip_unsupported_characters,
)


@pytest.fixture
def hindi_clean_sample() -> str:
    return "नमस्ते भारत! आप कैसे हैं?"


@pytest.fixture
def punjabi_clean_sample() -> str:
    return "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਜੀ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?"


@pytest.fixture
def diacritics_sample_hi() -> str:
    return "संसार में कर्म ही प्रधान है।"


@pytest.fixture
def diacritics_sample_pa() -> str:
    return "ਪੰਜਾਬੀ ਵਿੱਚ ਪਿੰਡ ਅਤੇ ਕੁੱਤਾ ਲਿਖੋ।"


@pytest.fixture
def dirty_emoji_sample() -> str:
    return "नमस्ते भारत! 😀 🔥 🎉 #speech @user https://example.com/audio %^&* आप कैसे हैं?"


def test_clean_samples_preserved(hindi_clean_sample, punjabi_clean_sample):
    """Verify clean sentences with allowed prosody remain unchanged."""
    assert normalize_text(hindi_clean_sample, "hi") == hindi_clean_sample
    assert normalize_text(punjabi_clean_sample, "pa") == punjabi_clean_sample


def test_phonetic_diacritics_preservation(diacritics_sample_hi, diacritics_sample_pa):
    """Verify virama, bindi, tippi, addak, chandrabindu, and danda are preserved."""
    res_hi = normalize_text(diacritics_sample_hi, "hi")
    assert res_hi == diacritics_sample_hi
    assert "\u094D" in res_hi  # virama
    assert "\u0902" in res_hi  # bindi
    assert "\u0964" in res_hi  # danda
    assert "\u0901" in normalize_text("माँ", "hi")  # chandrabindu

    res_pa = normalize_text(diacritics_sample_pa, "pa")
    assert res_pa == diacritics_sample_pa
    assert "\u0A70" in res_pa  # tippi
    assert "\u0A71" in res_pa  # addak
    assert "\u0964" in res_pa  # danda
    assert "\u0A02" in normalize_text("ਮੈਂ ਨਹੀਂ", "pa")  # bindi


def test_dirty_sample_cleaning(dirty_emoji_sample):
    """Verify emojis, Twitter tags, URLs, and forbidden symbols are stripped."""
    cleaned = normalize_text(dirty_emoji_sample, "hi")
    assert "😀" not in cleaned
    assert "🔥" not in cleaned
    assert "🎉" not in cleaned
    assert "#speech" not in cleaned
    assert "@user" not in cleaned
    assert "https://" not in cleaned
    assert "%" not in cleaned
    assert "^" not in cleaned
    assert "&" not in cleaned
    assert "*" not in cleaned
    assert cleaned == "नमस्ते भारत! आप कैसे हैं?"


@pytest.mark.parametrize(
    "number,expected",
    [
        (0, "शून्य"),
        (100, "एक सौ"),
        (105, "एक सौ पाँच"),
        (1000, "एक हज़ार"),
        (250000, "दो लाख पचास हज़ार"),
    ],
)
def test_number_to_words_hindi(number, expected):
    """Verify cardinal number conversion in Hindi."""
    assert number_to_words(number, "hi") == expected


@pytest.mark.parametrize(
    "number,expected",
    [
        (0, "ਸਿਫ਼ਰ"),
        (100, "ਇੱਕ ਸੌ"),
        (105, "ਇੱਕ ਸੌ ਪੰਜ"),
        (1000, "ਇੱਕ ਹਜ਼ਾਰ"),
        (250000, "ਦੋ ਲੱਖ ਪੰਜਾਹ ਹਜ਼ਾਰ"),
    ],
)
def test_number_to_words_punjabi(number, expected):
    """Verify cardinal number conversion in Punjabi."""
    assert number_to_words(number, "pa") == expected


def test_digit_expansion_sentences():
    """Verify digit expansion within realistic Hindi and Punjabi sentences."""
    hi_sent = "मेरे पास 100 रुपये हैं।"
    assert normalize_text(hi_sent, "hi") == "मेरे पास एक सौ रुपये हैं।"

    pa_sent = "ਮੇਰੇ ਕੋਲ 100 ਰੁਪਏ ਹਨ।"
    assert normalize_text(pa_sent, "pa") == "ਮੇਰੇ ਕੋਲ ਇੱਕ ਸੌ ਰੁਪਏ ਹਨ।"


def test_indic_native_digits():
    """Verify Devanagari and Gurmukhi script digits are parsed and expanded."""
    hi_indic_digits = "मेरे पास १०५ रुपये हैं।"
    assert normalize_text(hi_indic_digits, "hi") == "मेरे पास एक सौ पाँच रुपये हैं।"

    pa_indic_digits = "ਮੇਰੇ ਕੋਲ ੧੦੫ ਰੁਪਏ ਹਨ।"
    assert normalize_text(pa_indic_digits, "pa") == "ਮੇਰੇ ਕੋਲ ਇੱਕ ਸੌ ਪੰਜ ਰੁਪਏ ਹਨ।"


def test_normalizer_error_contracts():
    """Verify explicit error contracts for type and value validations."""
    with pytest.raises(ValueError, match="Unsupported language: en"):
        normalize_text("Hello", "en")

    with pytest.raises(ValueError, match="Unsupported language: fr"):
        number_to_words(10, "fr")

    with pytest.raises(TypeError, match="Expected text to be str"):
        normalize_text(123, "hi")  # type: ignore

    with pytest.raises(TypeError, match="Expected language to be str"):
        normalize_text("नमस्ते", 12)  # type: ignore

    with pytest.raises(ValueError, match="Negative numbers not supported: -5"):
        number_to_words(-5, "hi")

    with pytest.raises(TypeError, match="Expected number to be int"):
        number_to_words(12.34, "hi")  # type: ignore

    with pytest.raises(TypeError, match="Expected number to be int"):
        number_to_words(True, "hi")  # type: ignore


def test_indic_normalizer_class():
    """Verify object-oriented IndicNormalizer wrapper."""
    norm_hi = IndicNormalizer("hi")
    assert norm_hi.language == "hi"
    assert norm_hi.normalize("नमस्ते 100") == "नमस्ते एक सौ"

    norm_pa = IndicNormalizer("pa")
    assert norm_pa.language == "pa"
    assert norm_pa.normalize("ਸਤਿ 100") == "ਸਤਿ ਇੱਕ ਸੌ"

    with pytest.raises(ValueError, match="Unsupported language: de"):
        IndicNormalizer("de")

    with pytest.raises(TypeError, match="Expected language to be str"):
        IndicNormalizer(123)  # type: ignore
