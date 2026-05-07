"""Test fixtures for the Real-Time Voice Agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_agent.models import (
    AgentMetrics,
    AudioChunk,
    ConversationTurn,
    LLMResponse,
    ToolCall,
    ToolResult,
    TranscriptionResult,
    TTSConfig,
    VADConfig,
    VoiceAgentConfig,
)


@pytest.fixture
def sample_vad_config() -> VADConfig:
    """Create a sample VAD configuration."""
    return VADConfig(
        threshold=0.5,
        min_speech_duration=0.25,
        min_silence_duration=0.5,
        sample_rate=16000,
    )


@pytest.fixture
def sample_tts_config() -> TTSConfig:
    """Create a sample TTS configuration."""
    return TTSConfig(
        voice="en-US-AriaNeural",
        rate="+0%",
        volume="+0%",
        pitch="+0Hz",
    )


@pytest.fixture
def sample_agent_config(sample_vad_config, sample_tts_config) -> VoiceAgentConfig:
    """Create a sample voice agent configuration."""
    return VoiceAgentConfig(
        model_name="gemini-2.5-flash",
        vad=sample_vad_config,
        tts=sample_tts_config,
        system_prompt="You are a helpful voice assistant.",
        max_conversation_turns=20,
        interrupt_enabled=True,
    )


@pytest.fixture
def sample_audio_chunk() -> AudioChunk:
    """Create a sample audio chunk."""
    # Generate 1 second of silence
    samples = np.zeros(16000, dtype=np.int16)
    return AudioChunk(
        data=samples.tobytes(),
        sample_rate=16000,
        channels=1,
    )


@pytest.fixture
def sample_speech_audio_chunk() -> AudioChunk:
    """Create a sample audio chunk with speech-like content."""
    # Generate some "speech-like" audio (sine wave with noise)
    t = np.linspace(0, 1, 16000)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(16000)
    samples = (signal * 32767).astype(np.int16)
    return AudioChunk(
        data=samples.tobytes(),
        sample_rate=16000,
        channels=1,
    )


@pytest.fixture
def sample_conversation_turn() -> ConversationTurn:
    """Create a sample conversation turn."""
    return ConversationTurn(
        role="user",
        content="Hello, how are you?",
        audio_duration=1.5,
    )


@pytest.fixture
def sample_transcription_result() -> TranscriptionResult:
    """Create a sample transcription result."""
    return TranscriptionResult(
        text="Hello, this is a test.",
        confidence=0.95,
        language="en",
        duration=2.0,
    )


@pytest.fixture
def sample_llm_response() -> LLMResponse:
    """Create a sample LLM response."""
    return LLMResponse(
        text="Hello! I'm doing well, thank you for asking.",
        tool_calls=[],
        finish_reason="stop",
    )


@pytest.fixture
def sample_llm_response_with_tool() -> LLMResponse:
    """Create a sample LLM response with tool call."""
    return LLMResponse(
        text="Let me check the weather for you. [TOOL:get_weather(city=London)]",
        tool_calls=[ToolCall(name="get_weather", arguments={"city": "London"})],
        finish_reason="stop",
    )


@pytest.fixture
def sample_tool_result() -> ToolResult:
    """Create a sample tool result."""
    return ToolResult(
        name="get_weather",
        result={"temperature": 15, "conditions": "cloudy"},
        success=True,
    )


@pytest.fixture
def sample_agent_metrics() -> AgentMetrics:
    """Create sample agent metrics."""
    return AgentMetrics(
        total_turns=10,
        avg_response_time=250.0,
        avg_transcription_time=100.0,
        avg_tts_time=150.0,
        total_audio_duration=30.0,
        error_count=0,
    )


@pytest.fixture
def mock_gemini_model():
    """Create a mock Gemini model."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello! How can I help you today?"
    mock_response.candidates = [MagicMock(finish_reason="STOP")]
    mock.generate_content.return_value = mock_response
    return mock


@pytest.fixture
def mock_tts():
    """Create a mock TTS."""
    mock = AsyncMock()
    mock.synthesize.return_value = b"\x00" * 1000  # Mock audio bytes
    return mock


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("TTS_VOICE", "en-US-AriaNeural")
    monkeypatch.setenv("VAD_THRESHOLD", "0.5")
