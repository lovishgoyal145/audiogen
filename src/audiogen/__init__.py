"""AudioGen: High-performance audio generation and operational service platform."""

from audiogen.audio_processor import AudioProcessor
from audiogen.config import Settings, get_settings
from audiogen.engine import Synthesizer
from audiogen.normalizer import IndicNormalizer, normalize_text

__version__ = "0.1.0"

__all__ = [
    "AudioProcessor",
    "IndicNormalizer",
    "Settings",
    "Synthesizer",
    "get_settings",
    "normalize_text",
]
