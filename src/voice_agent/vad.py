"""Voice Activity Detection using Silero VAD.

This module provides a wrapper around Silero VAD for detecting speech
in audio streams with low latency.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import suppress
from importlib import import_module
from typing import Any

import numpy as np
from dotenv import load_dotenv

from voice_agent.models import AudioChunk, VADConfig

load_dotenv()

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Silero VAD wrapper for speech detection.

    Silero VAD is a lightweight, open-source voice activity detector
    that runs efficiently on CPU.
    """

    def __init__(self, config: VADConfig | None = None):
        """Initialize the VAD.

        Args:
            config: VAD configuration. Uses defaults if not provided.
        """
        self.config = config or VADConfig(
            threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
            min_speech_duration=float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.25")),
            min_silence_duration=float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.5")),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
        )
        self._model: Any | None = None
        self._utils: Any | None = None

    def _load_model(self) -> None:
        """Lazy load the Silero VAD model."""
        if self._model is None:
            try:
                torch = import_module("torch")

                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self._model = model
                self._utils = utils
            except Exception as e:
                logger.warning("Falling back to energy-based VAD: %s", e)
                self._model = "fallback"

    def detect_speech(self, audio: np.ndarray) -> bool:
        """Detect if speech is present in audio.

        Args:
            audio: Audio samples as numpy array (float32, -1 to 1).

        Returns:
            True if speech is detected, False otherwise.
        """
        self._load_model()

        if self._model == "fallback":
            return self._fallback_detect(audio)

        model = self._model
        if model is None:
            return self._fallback_detect(audio)

        try:
            torch = import_module("torch")

            # Ensure correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio)

            # Get speech probability
            speech_prob = model(audio_tensor, self.config.sample_rate).item()

            return speech_prob >= self.config.threshold

        except Exception as e:
            logger.exception("VAD error: %s", e)
            return self._fallback_detect(audio)

    def _fallback_detect(self, audio: np.ndarray) -> bool:
        """Fallback energy-based speech detection.

        Args:
            audio: Audio samples.

        Returns:
            True if energy exceeds threshold.
        """
        # Simple RMS energy threshold
        rms = np.sqrt(np.mean(audio**2))
        return rms > 0.02  # Adjust threshold as needed

    def process_stream(
        self,
        audio_iterator: Iterator[AudioChunk],
        chunk_duration: float = 0.03,
    ) -> Iterator[tuple[AudioChunk, bool]]:
        """Process an audio stream and yield chunks with speech detection.

        Args:
            audio_iterator: Iterator of audio chunks.
            chunk_duration: Duration of each chunk in seconds.

        Yields:
            Tuples of (audio_chunk, is_speech).
        """
        speech_buffer: list[AudioChunk] = []
        silence_duration = 0.0
        speech_duration = 0.0
        is_speaking = False

        for chunk in audio_iterator:
            # Convert bytes to numpy array
            audio = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0

            has_speech = self.detect_speech(audio)

            if has_speech:
                speech_duration += chunk_duration
                silence_duration = 0.0

                if not is_speaking and speech_duration >= self.config.min_speech_duration:
                    is_speaking = True

                if is_speaking:
                    speech_buffer.append(chunk)
                    yield chunk, True

            else:
                if is_speaking:
                    silence_duration += chunk_duration

                    if silence_duration >= self.config.min_silence_duration:
                        # End of speech
                        is_speaking = False
                        speech_duration = 0.0
                        speech_buffer.clear()
                        yield chunk, False
                    else:
                        # Brief pause, continue buffering
                        speech_buffer.append(chunk)
                        yield chunk, True
                else:
                    speech_duration = 0.0
                    yield chunk, False

    def reset(self) -> None:
        """Reset the VAD state."""
        if self._model is not None and self._model != "fallback":
            with suppress(Exception):
                self._model.reset_states()


def create_vad(config: VADConfig | None = None) -> VoiceActivityDetector:
    """Create a VoiceActivityDetector instance.

    Args:
        config: Optional VAD configuration.

    Returns:
        VoiceActivityDetector instance.
    """
    return VoiceActivityDetector(config)
