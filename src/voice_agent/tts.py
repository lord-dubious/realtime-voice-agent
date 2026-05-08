"""Text-to-Speech using Edge TTS.

This module provides TTS functionality using Microsoft Edge's free
TTS service for high-quality voice synthesis.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from dotenv import load_dotenv

from voice_agent.models import TTSConfig

load_dotenv()

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Edge TTS wrapper for text-to-speech synthesis.

    Uses Microsoft Edge's free TTS service which provides high-quality
    neural voices without API costs.
    """

    def __init__(self, config: TTSConfig | None = None):
        """Initialize TTS.

        Args:
            config: TTS configuration. Uses defaults if not provided.
        """
        self.config = config or TTSConfig(
            voice=os.getenv("TTS_VOICE", "en-US-AriaNeural"),
        )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.

        Returns:
            Audio data as bytes (MP3 format).
        """
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self.config.rate,
                volume=self.config.volume,
                pitch=self.config.pitch,
            )

            audio_data = io.BytesIO()

            async for chunk in communicate.stream():
                data = chunk.get("data")
                if chunk["type"] == "audio" and isinstance(data, bytes):
                    audio_data.write(data)

            return audio_data.getvalue()

        except Exception as e:
            logger.exception("TTS synthesis error: %s", e)
            return b""

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesized speech.

        Args:
            text: Text to synthesize.

        Yields:
            Audio chunks as bytes.
        """
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self.config.rate,
                volume=self.config.volume,
                pitch=self.config.pitch,
            )

            async for chunk in communicate.stream():
                data = chunk.get("data")
                if chunk["type"] == "audio" and isinstance(data, bytes):
                    yield data

        except Exception as e:
            logger.exception("TTS streaming error: %s", e)

    @staticmethod
    async def list_voices(language: str = "en") -> list[dict[str, Any]]:
        """List available voices for a language.

        Args:
            language: Language code to filter voices.

        Returns:
            List of voice info dictionaries.
        """
        try:
            import edge_tts

            voices = await edge_tts.list_voices()
            return [dict(v) for v in voices if v["Locale"].startswith(language)]

        except Exception as e:
            logger.exception("Failed to list voices: %s", e)
            return []


def create_tts(config: TTSConfig | None = None) -> TextToSpeech:
    """Create a TextToSpeech instance.

    Args:
        config: Optional TTS configuration.

    Returns:
        TextToSpeech instance.
    """
    return TextToSpeech(config)
