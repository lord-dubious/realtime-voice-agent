"""Tests for Text-to-Speech."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_agent.models import TTSConfig
from voice_agent.tts import TextToSpeech, create_tts


class TestTextToSpeech:
    """Tests for TextToSpeech class."""

    def test_init_default_config(self):
        """Test TTS initialization with default config."""
        tts = TextToSpeech()
        assert tts.config.voice == "en-US-AriaNeural"

    def test_init_custom_config(self, sample_tts_config):
        """Test TTS initialization with custom config."""
        tts = TextToSpeech(config=sample_tts_config)
        assert tts.config.voice == "en-US-AriaNeural"
        assert tts.config.rate == "+0%"

    @pytest.mark.asyncio
    async def test_synthesize_success(self):
        """Test successful speech synthesis."""
        tts = TextToSpeech()

        mock_communicate = MagicMock()

        async def mock_stream():
            yield {"type": "audio", "data": b"\x00\x01\x02"}
            yield {"type": "audio", "data": b"\x03\x04\x05"}

        mock_communicate.stream = mock_stream

        with patch("edge_tts.Communicate", return_value=mock_communicate):
            result = await tts.synthesize("Hello world")

            assert result == b"\x00\x01\x02\x03\x04\x05"

    @pytest.mark.asyncio
    async def test_synthesize_handles_non_audio_chunks(self):
        """Test synthesis handles non-audio chunk types."""
        tts = TextToSpeech()

        mock_communicate = MagicMock()

        async def mock_stream():
            yield {"type": "metadata", "data": "info"}
            yield {"type": "audio", "data": b"\x00\x01"}
            yield {"type": "end", "data": None}

        mock_communicate.stream = mock_stream

        with patch("edge_tts.Communicate", return_value=mock_communicate):
            result = await tts.synthesize("Test")

            assert result == b"\x00\x01"

    @pytest.mark.asyncio
    async def test_synthesize_error_handling(self, caplog, capsys):
        """Test synthesis error handling logs instead of printing."""
        tts = TextToSpeech()

        with (
            patch("edge_tts.Communicate", side_effect=Exception("TTS error")),
            caplog.at_level("ERROR", logger="voice_agent.tts"),
        ):
            result = await tts.synthesize("Test")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "TTS synthesis error" in caplog.text
        assert result == b""

    @pytest.mark.asyncio
    async def test_synthesize_stream(self):
        """Test streaming synthesis."""
        tts = TextToSpeech()

        mock_communicate = MagicMock()

        async def mock_stream():
            yield {"type": "audio", "data": b"chunk1"}
            yield {"type": "audio", "data": b"chunk2"}

        mock_communicate.stream = mock_stream

        with patch("edge_tts.Communicate", return_value=mock_communicate):
            chunks = []
            async for chunk in tts.synthesize_stream("Hello"):
                chunks.append(chunk)

            assert chunks == [b"chunk1", b"chunk2"]

    @pytest.mark.asyncio
    async def test_synthesize_stream_error(self, caplog, capsys):
        """Test streaming synthesis error handling logs instead of printing."""
        tts = TextToSpeech()

        with (
            patch("edge_tts.Communicate", side_effect=Exception("Stream error")),
            caplog.at_level("ERROR", logger="voice_agent.tts"),
        ):
            chunks = []
            async for chunk in tts.synthesize_stream("Test"):
                chunks.append(chunk)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "TTS streaming error" in caplog.text
        # Should yield nothing on error
        assert chunks == []

    @pytest.mark.asyncio
    async def test_list_voices_success(self):
        """Test listing available voices."""
        mock_voices = [
            {"ShortName": "en-US-AriaNeural", "Locale": "en-US", "Gender": "Female"},
            {"ShortName": "en-US-GuyNeural", "Locale": "en-US", "Gender": "Male"},
            {"ShortName": "de-DE-KatjaNeural", "Locale": "de-DE", "Gender": "Female"},
        ]

        with patch("edge_tts.list_voices", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_voices

            voices = await TextToSpeech.list_voices("en")

            assert len(voices) == 2  # Only English voices
            assert all(v["Locale"].startswith("en") for v in voices)

    @pytest.mark.asyncio
    async def test_list_voices_error(self, caplog, capsys):
        """Test listing voices error handling logs instead of printing."""
        with (
            patch("edge_tts.list_voices", side_effect=Exception("API error")),
            caplog.at_level("ERROR", logger="voice_agent.tts"),
        ):
            voices = await TextToSpeech.list_voices("en")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Failed to list voices" in caplog.text
        assert voices == []


class TestCreateTts:
    """Tests for create_tts factory function."""

    def test_create_tts_default(self):
        """Test creating TTS with defaults."""
        tts = create_tts()
        assert isinstance(tts, TextToSpeech)

    def test_create_tts_with_config(self, sample_tts_config):
        """Test creating TTS with custom config."""
        tts = create_tts(sample_tts_config)
        assert tts.config == sample_tts_config


class TestTTSConfiguration:
    """Tests for TTS configuration options."""

    def test_custom_voice(self):
        """Test using a custom voice."""
        config = TTSConfig(voice="en-GB-SoniaNeural")
        tts = TextToSpeech(config=config)

        assert tts.config.voice == "en-GB-SoniaNeural"

    def test_rate_adjustment(self):
        """Test speech rate adjustment."""
        config = TTSConfig(rate="+25%")
        tts = TextToSpeech(config=config)

        assert tts.config.rate == "+25%"

    def test_volume_adjustment(self):
        """Test volume adjustment."""
        config = TTSConfig(volume="-10%")
        tts = TextToSpeech(config=config)

        assert tts.config.volume == "-10%"

    def test_pitch_adjustment(self):
        """Test pitch adjustment."""
        config = TTSConfig(pitch="+5Hz")
        tts = TextToSpeech(config=config)

        assert tts.config.pitch == "+5Hz"
