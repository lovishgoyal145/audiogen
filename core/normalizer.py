"""Indic text normalization module for Hindi (hi) and Punjabi (pa)."""

from typing import Final, Optional, Set
import re

SUPPORTED_LANGUAGES: Final[Set[str]] = {"hi", "pa"}

HINDI_0_TO_99 = {
    0: "शून्य",
    1: "एक",
    2: "दो",
    3: "तीन",
    4: "चार",
    5: "पाँच",
    6: "छह",
    7: "सात",
    8: "आठ",
    9: "नौ",
    10: "दस",
    11: "ग्यारह",
    12: "बारह",
    13: "तेरह",
    14: "चौदह",
    15: "पंद्रह",
    16: "सोलह",
    17: "सत्रह",
    18: "अठारह",
    19: "उन्नीस",
    20: "बीस",
    21: "इक्कीस",
    22: "बाईस",
    23: "तेईस",
    24: "चौबीस",
    25: "पच्चीस",
    26: "छब्बीस",
    27: "सत्ताईस",
    28: "अट्ठाईस",
    29: "उनतीस",
    30: "तीस",
    31: "इकतीस",
    32: "बत्तीस",
    33: "तैंतीस",
    34: "चौंतीस",
    35: "पैंतीस",
    36: "छत्तीस",
    37: "सैंतीस",
    38: "अड़तीस",
    39: "उनतालीस",
    40: "चालीस",
    41: "इकतालीस",
    42: "बयालीस",
    43: "तैंतालीस",
    44: "चवालीस",
    45: "पैंतालीस",
    46: "छियालीस",
    47: "सैंतालीस",
    48: "अड़तालीस",
    49: "उनचास",
    50: "पचास",
    51: "इक्यावन",
    52: "बावन",
    53: "तिरपन",
    54: "चौवन",
    55: "पचपन",
    56: "छप्पन",
    57: "सत्तावन",
    58: "अट्ठावन",
    59: "उनसठ",
    60: "साठ",
    61: "इकसठ",
    62: "बासठ",
    63: "तिरसठ",
    64: "चौंसठ",
    65: "पैंसठ",
    66: "छियासठ",
    67: "सरसठ",
    68: "अड़सठ",
    69: "उनहत्तर",
    70: "सत्तर",
    71: "इकहत्तर",
    72: "बहत्तर",
    73: "तिहत्तर",
    74: "चौहत्तर",
    75: "पचहत्तर",
    76: "छिहत्तर",
    77: "सतहत्तर",
    78: "अठहत्तर",
    79: "उन्यासी",
    80: "अस्सी",
    81: "इक्यासी",
    82: "बयासी",
    83: "तिरयासी",
    84: "चौरासी",
    85: "पचासी",
    86: "छियासी",
    87: "सत्तासी",
    88: "अट्ठासी",
    89: "नवासी",
    90: "नब्बे",
    91: "इक्यानवे",
    92: "बानवे",
    93: "तिरानवे",
    94: "चौरानवे",
    95: "पचानवे",
    96: "छानवे",
    97: "सत्तानवे",
    98: "अट्ठानवे",
    99: "निन्यानवे",
}

PUNJABI_0_TO_99 = {
    0: "ਸਿਫ਼ਰ",
    1: "ਇੱਕ",
    2: "ਦੋ",
    3: "ਤਿੰਨ",
    4: "ਚਾਰ",
    5: "ਪੰਜ",
    6: "ਛੇ",
    7: "ਸੱਤ",
    8: "ਅੱਠ",
    9: "ਨੌਂ",
    10: "ਦਸ",
    11: "ਗਿਆਰਾਂ",
    12: "ਬਾਰਾਂ",
    13: "ਤੇਰਾਂ",
    14: "ਚੌਦਾਂ",
    15: "ਪੰਦਰਾਂ",
    16: "ਸੋਲਾਂ",
    17: "ਸਤਾਰਾਂ",
    18: "ਅਠਾਰਾਂ",
    19: "ਉੱਨੀ",
    20: "ਵੀਹ",
    21: "ਇੱਕੀ",
    22: "ਬਾਈ",
    23: "ਤੇਈ",
    24: "ਚੌਵੀ",
    25: "ਪੱਚੀ",
    26: "ਛੱਬੀ",
    27: "ਸਤਾਈ",
    28: "ਅਠਾਈ",
    29: "ਉਣੱਤੀ",
    30: "ਤੀਹ",
    31: "ਇਕੱਤੀ",
    32: "ਬੱਤੀ",
    33: "ਤੇਤੀ",
    34: "ਚੌਂਤੀ",
    35: "ਪੈਂਤੀ",
    36: "ਛੱਤੀ",
    37: "ਸੈਂਤੀ",
    38: "ਅਠੱਤੀ",
    39: "ਉਣਤਾਲੀ",
    40: "ਚਾਲੀ",
    41: "ਇਕਤਾਲੀ",
    42: "ਬਤਾਲੀ",
    43: "ਤਰਤਾਲੀ",
    44: "ਚੁਤਾਲੀ",
    45: "ਪੈਤਾਲੀ",
    46: "ਛਿਆਲੀ",
    47: "ਸੰਤਾਲੀ",
    48: "ਅਠਤਾਲੀ",
    49: "ਉਣੰਜਾ",
    50: "ਪੰਜਾਹ",
    51: "ਇਕਵੰਜਾ",
    52: "ਬਵੰਜਾ",
    53: "ਤਿਰਵੰਜਾ",
    54: "ਚਵੰਜਾ",
    55: "ਪਚਵੰਜਾ",
    56: "ਛਪੰਜਾ",
    57: "ਸਤਵੰਜਾ",
    58: "ਅਠਵੰਜਾ",
    59: "ਉਣਾਹਠ",
    60: "ਸੱਠ",
    61: "ਇਕਾਹਠ",
    62: "ਬਾਹਠ",
    63: "ਤ੍ਰੇਹਠ",
    64: "ਚੌਂਹਠ",
    65: "ਪੈਂਠ",
    66: "ਛਿਆਹਠ",
    67: "ਸਤਾਹਠ",
    68: "ਅਠਾਹਠ",
    69: "ਉਣੱਤਰ",
    70: "ਸੱਤਰ",
    71: "ਇਕੱਤਰ",
    72: "ਬਹੱਤਰ",
    73: "ਤਿਹੱਤਰ",
    74: "ਚੌਹੱਤਰ",
    75: "ਪਚੱਤਰ",
    76: "ਛਿਹੱਤਰ",
    77: "ਸਤੱਤਰ",
    78: "ਅਠੱਤਰ",
    79: "ਉਨਾਸੀ",
    80: "ਅੱਸੀ",
    81: "ਇਕਿਆਸੀ",
    82: "ਬਿਆਸੀ",
    83: "ਤਿਰਾਸੀ",
    84: "ਚੌਰਾਸੀ",
    85: "ਪਚਾਸੀ",
    86: "ਛਿਆਸੀ",
    87: "ਸਤਾਸੀ",
    88: "ਅਠਾਸੀ",
    89: "ਉਣਾਨਵੇਂ",
    90: "ਨੱਬੇ",
    91: "ਇਕਾਨਵੇਂ",
    92: "ਬਾਨਵੇਂ",
    93: "ਤਿਰਾਨਵੇਂ",
    94: "ਚੌਰਾਨਵੇਂ",
    95: "ਪਚਾਨਵੇਂ",
    96: "ਛਿਆਨਵੇਂ",
    97: "ਸਤਾਨਵੇਂ",
    98: "ਅਠਾਨਵੇਂ",
    99: "ਨੜਿੰਨਵੇਂ",
}

# Pre-compute allowed character sets for fast lookup
_HI_ALLOWED: Final[Set[str]] = set(" ,?!।॥\t\n")
for _cp in range(0x0900, 0x0980):
    _HI_ALLOWED.add(chr(_cp))

_PA_ALLOWED: Final[Set[str]] = set(" ,?!।॥\t\n")
for _cp in range(0x0A00, 0x0A80):
    _PA_ALLOWED.add(chr(_cp))
_PA_ALLOWED.add("\u0964")  # danda
_PA_ALLOWED.add("\u0965")  # double danda

_DIGIT_PATTERN: Final[re.Pattern] = re.compile(r"[0-9\u0966-\u096F\u0A66-\u0A6F]+")
_URL_PATTERN: Final[re.Pattern] = re.compile(r"https?://\S+|www\.\S+")
_TAG_PATTERN: Final[re.Pattern] = re.compile(r"[@#][\w\u0900-\u097F\u0A00-\u0A7F]+")


def number_to_words(number: int, language: str) -> str:
    """Convert an integer to its spelled-out Indic word representation.

    Supports numbers from 0 to 99,99,99,999 (crores) following Indic numbering.

    Args:
        number: Non-negative integer to convert.
        language: Target language code ('hi' for Hindi, 'pa' for Punjabi).

    Returns:
        Word representation in Devanagari or Gurmukhi script.

    Raises:
        ValueError: If number < 0 or language is unsupported.
        TypeError: If number is not an integer or language is not a string.
    """
    if not isinstance(number, int) or isinstance(number, bool):
        raise TypeError(f"Expected number to be int, got {type(number).__name__}")
    if not isinstance(language, str):
        raise TypeError(f"Expected language to be str, got {type(language).__name__}")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    if number < 0:
        raise ValueError(f"Negative numbers not supported: {number}")

    if number == 0:
        return "शून्य" if language == "hi" else "ਸਿਫ਼ਰ"

    table_0_to_99 = HINDI_0_TO_99 if language == "hi" else PUNJABI_0_TO_99
    crore_word = "करोड़" if language == "hi" else "ਕਰੋੜ"
    lakh_word = "लाख" if language == "hi" else "ਲੱਖ"
    thousand_word = "हज़ार" if language == "hi" else "ਹਜ਼ਾਰ"
    hundred_word = "सौ" if language == "hi" else "ਸੌ"

    parts = []

    # Crores (10,000,000)
    crores = number // 10_000_000
    remainder = number % 10_000_000
    if crores > 0:
        parts.append(f"{number_to_words(crores, language)} {crore_word}")

    # Lakhs (100,000)
    lakhs = remainder // 100_000
    remainder = remainder % 100_000
    if lakhs > 0:
        parts.append(f"{table_0_to_99[lakhs]} {lakh_word}")

    # Thousands (1,000)
    thousands = remainder // 1_000
    remainder = remainder % 1_000
    if thousands > 0:
        parts.append(f"{table_0_to_99[thousands]} {thousand_word}")

    # Hundreds (100)
    hundreds = remainder // 100
    remainder = remainder % 100
    if hundreds > 0:
        parts.append(f"{table_0_to_99[hundreds]} {hundred_word}")

    # Remainder (1-99)
    if remainder > 0:
        parts.append(table_0_to_99[remainder])

    return " ".join(parts)


def expand_digits(text: str, language: str) -> str:
    """Detect all numeric tokens (ASCII '0-9', Devanagari '०-९', Gurmukhi '੦-੯')
    and expand them into spoken words in the target language script.

    Args:
        text: Input text containing numeric digits.
        language: Target language code ('hi' or 'pa').

    Returns:
        Text with all digits converted to spelled-out spoken words.

    Raises:
        ValueError: If language is not in {'hi', 'pa'}.
        TypeError: If text is not a string or language is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    if not isinstance(language, str):
        raise TypeError(f"Expected language to be str, got {type(language).__name__}")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    def _replace_match(match: re.Match) -> str:
        token = match.group(0)
        digits = []
        for c in token:
            if "0" <= c <= "9":
                digits.append(ord(c) - ord("0"))
            elif "\u0966" <= c <= "\u096F":
                digits.append(ord(c) - 0x0966)
            elif "\u0A66" <= c <= "\u0A6F":
                digits.append(ord(c) - 0x0A66)
            else:
                digits.append(0)
        val = int("".join(str(d) for d in digits))
        return number_to_words(val, language)

    return _DIGIT_PATTERN.sub(_replace_match, text)


def strip_unsupported_characters(text: str, language: str) -> str:
    """Strip emojis, control codes, and symbols outside the allowed phonetic alphabet.

    Preserves:
    - Hindi: Devanagari letters, matras, nukta (\u093C), virama (\u094D),
             bindi/anusvara (\u0902), chandrabindu (\u0901), danda (\u0964), double danda (\u0965).
    - Punjabi: Gurmukhi letters, matras, nukta (\u0A3C), virama (\u0A4D),
               bindi (\u0A02), tippi (\u0A70), addak (\u0A71), danda (\u0964), double danda (\u0965).
    - Permitted prosodic punctuation: whitespace, comma (,), question mark (?), exclamation (!).

    Args:
        text: Raw text.
        language: Language code ('hi' or 'pa').

    Returns:
        Cleaned text string containing only valid characters.

    Raises:
        ValueError: If language is not in {'hi', 'pa'}.
        TypeError: If text is not a string or language is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    if not isinstance(language, str):
        raise TypeError(f"Expected language to be str, got {type(language).__name__}")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    allowed = _HI_ALLOWED if language == "hi" else _PA_ALLOWED
    return "".join(c for c in text if c in allowed)


def normalize_text(text: str, language: str) -> str:
    """Normalize raw text for Indic TTS synthesis.

    Strips emojis, URLs, and unsupported symbols while preserving required diacritics
    (virama/halant, bindi, tippi, addak, danda) and expanding digits into spoken words.

    Args:
        text: Raw input string in Devanagari, Gurmukhi, or mixed script with digits.
        language: ISO language code ('hi' or 'pa').

    Returns:
        Cleaned, normalized string with spoken word representations for all numerals.

    Raises:
        ValueError: If language is not in {'hi', 'pa'}.
        TypeError: If text is not a string or language is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    if not isinstance(language, str):
        raise TypeError(f"Expected language to be str, got {type(language).__name__}")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    # 1. Strip URLs
    cleaned = _URL_PATTERN.sub("", text)

    # 2. Strip Twitter mentions and hashtags
    cleaned = _TAG_PATTERN.sub("", cleaned)

    # 3. Expand numeric digits into spoken words
    cleaned = expand_digits(cleaned, language)

    # 4. Strip unsupported characters, emojis, math symbols
    cleaned = strip_unsupported_characters(cleaned, language)

    # 5. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


class IndicNormalizer:
    """Stateful or object-oriented interface for Indic text normalization."""

    def __init__(self, language: str) -> None:
        """Initialize normalizer for a specific language.

        Args:
            language: Language code ('hi' or 'pa').

        Raises:
            ValueError: If language is not supported.
            TypeError: If language is not a string.
        """
        if not isinstance(language, str):
            raise TypeError(f"Expected language to be str, got {type(language).__name__}")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        self._language = language

    @property
    def language(self) -> str:
        """Return the configured language code."""
        return self._language

    def normalize(self, text: str) -> str:
        """Normalize input text using the configured language."""
        return normalize_text(text, self._language)
