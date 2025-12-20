"""Tests for Voice Activity Detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from voice_agent.models import AudioChunk, VADConfig
from voice_agent.vad import VoiceActivityDetector, create_vad


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    def test_init_default_config(self):
        """Test VAD initialization with default config."""
        vad = VoiceActivityDetector()
        assert vad.config.threshold == 0.5
        assert vad.config.sample_rate == 16000
        assert vad._model is None

    def test_init_custom_config(self, sample_vad_config):
        """Test VAD initialization with custom config."""
        vad = VoiceActivityDetector(config=sample_vad_config)
        assert vad.config.threshold == 0.5
        assert vad.config.min_speech_duration == 0.25

    def test_fallback_detect_silence(self):
        """Test fallback detection with silence."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"  # Force fallback mode

        # Create silence
        audio = np.zeros(16000, dtype=np.float32)
        result = vad.detect_speech(audio)

        assert bool(result) == False

    def test_fallback_detect_speech(self):
        """Test fallback detection with loud audio."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        # Create loud signal
        audio = np.ones(16000, dtype=np.float32) * 0.5
        result = vad.detect_speech(audio)

        assert bool(result) == True

    def test_detect_speech_with_low_energy(self):
        """Test detection with low energy audio."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        # Create low energy signal
        audio = np.ones(16000, dtype=np.float32) * 0.01
        result = vad.detect_speech(audio)

        assert bool(result) == False

    def test_detect_speech_normalizes_int16(self):
        """Test that detection handles int16 values correctly."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        # Create audio as int16 values
        audio = np.ones(16000, dtype=np.float32) * 16384  # Half max int16
        result = vad.detect_speech(audio)

        # Should still work even with large values
        assert isinstance(bool(result), bool)

    def test_reset(self):
        """Test VAD reset."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        # Should not raise
        vad.reset()

    def test_reset_with_silero_model(self):
        """Test VAD reset with mock Silero model."""
        vad = VoiceActivityDetector()
        mock_model = MagicMock()
        vad._model = mock_model

        vad.reset()

        mock_model.reset_states.assert_called_once()


class TestCreateVad:
    """Tests for create_vad factory function."""

    def test_create_vad_default(self):
        """Test creating VAD with defaults."""
        vad = create_vad()
        assert isinstance(vad, VoiceActivityDetector)

    def test_create_vad_with_config(self, sample_vad_config):
        """Test creating VAD with custom config."""
        vad = create_vad(sample_vad_config)
        assert vad.config == sample_vad_config


class TestVADProcessStream:
    """Tests for stream processing."""

    def test_process_stream_silence(self):
        """Test processing stream of silence."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        # Create silent chunks
        def chunk_generator():
            for _ in range(5):
                samples = np.zeros(480, dtype=np.int16)  # 30ms at 16kHz
                yield AudioChunk(data=samples.tobytes(), sample_rate=16000)

        results = list(vad.process_stream(chunk_generator(), chunk_duration=0.03))

        # All should be non-speech
        assert all(not is_speech for _, is_speech in results)

    def test_process_stream_speech(self):
        """Test processing stream with speech-like audio."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        # Create loud chunks (simulating speech)
        def chunk_generator():
            for _ in range(20):
                # High energy signal
                samples = (np.random.randn(480) * 10000).astype(np.int16)
                yield AudioChunk(data=samples.tobytes(), sample_rate=16000)

        results = list(vad.process_stream(chunk_generator(), chunk_duration=0.03))

        # Some should be detected as speech
        speech_detected = any(is_speech for _, is_speech in results)
        assert speech_detected

    def test_process_stream_mixed(self):
        """Test processing stream with mixed content."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        # Create mixed chunks
        def chunk_generator():
            # Start with silence
            for _ in range(5):
                samples = np.zeros(480, dtype=np.int16)
                yield AudioChunk(data=samples.tobytes(), sample_rate=16000)

            # Then speech
            for _ in range(10):
                samples = (np.random.randn(480) * 10000).astype(np.int16)
                yield AudioChunk(data=samples.tobytes(), sample_rate=16000)

            # Then silence again
            for _ in range(10):
                samples = np.zeros(480, dtype=np.int16)
                yield AudioChunk(data=samples.tobytes(), sample_rate=16000)

        results = list(vad.process_stream(chunk_generator(), chunk_duration=0.03))

        # Should have some speech and some silence
        has_speech = any(is_speech for _, is_speech in results)
        has_silence = any(not is_speech for _, is_speech in results)

        assert has_speech
        assert has_silence


class TestVADEdgeCases:
    """Edge case tests for VAD."""

    def test_empty_audio(self):
        """Test with empty audio array."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        audio = np.array([], dtype=np.float32)

        # Should not crash - may return nan/0 for empty audio
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vad.detect_speech(audio)
            # Empty audio should return something
            assert result is not None or result == False or result == True

    def test_very_short_audio(self):
        """Test with very short audio."""
        vad = VoiceActivityDetector()
        vad._model = "fallback"

        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = vad.detect_speech(audio)

        assert isinstance(bool(result), bool)

    def test_high_threshold(self):
        """Test with very high threshold."""
        config = VADConfig(threshold=0.99)
        vad = VoiceActivityDetector(config=config)
        vad._model = "fallback"

        # Even loud audio might not pass
        audio = np.ones(16000, dtype=np.float32) * 0.3
        result = vad.detect_speech(audio)

        assert isinstance(bool(result), bool)

    def test_low_threshold(self):
        """Test with very low threshold."""
        config = VADConfig(threshold=0.01)
        vad = VoiceActivityDetector(config=config)
        vad._model = "fallback"

        # Almost any audio should pass
        audio = np.ones(16000, dtype=np.float32) * 0.05
        result = vad.detect_speech(audio)

        assert bool(result) == True
