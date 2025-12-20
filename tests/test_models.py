"""Tests for Pydantic models."""

from __future__ import annotations

from datetime import datetime

import pytest

from voice_agent.models import (
    AgentMetrics,
    AgentState,
    AudioChunk,
    ConversationTurn,
    LLMResponse,
    RoomConfig,
    ToolCall,
    ToolResult,
    TranscriptionResult,
    TTSConfig,
    VADConfig,
    VoiceAgentConfig,
)


class TestAgentState:
    """Tests for AgentState enum."""

    def test_agent_states(self):
        """Test agent state values."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.LISTENING.value == "listening"
        assert AgentState.PROCESSING.value == "processing"
        assert AgentState.SPEAKING.value == "speaking"
        assert AgentState.ERROR.value == "error"


class TestVADConfig:
    """Tests for VADConfig model."""

    def test_create_vad_config(self):
        """Test creating VAD config."""
        config = VADConfig(
            threshold=0.6,
            min_speech_duration=0.3,
            min_silence_duration=0.4,
            sample_rate=16000,
        )
        assert config.threshold == 0.6
        assert config.min_speech_duration == 0.3
        assert config.sample_rate == 16000

    def test_vad_config_defaults(self):
        """Test VAD config default values."""
        config = VADConfig()
        assert config.threshold == 0.5
        assert config.min_speech_duration == 0.25
        assert config.min_silence_duration == 0.5
        assert config.sample_rate == 16000

    def test_vad_threshold_bounds(self):
        """Test VAD threshold validation."""
        config = VADConfig(threshold=0.0)
        assert config.threshold == 0.0

        config = VADConfig(threshold=1.0)
        assert config.threshold == 1.0


class TestTTSConfig:
    """Tests for TTSConfig model."""

    def test_create_tts_config(self):
        """Test creating TTS config."""
        config = TTSConfig(
            voice="en-GB-RyanNeural",
            rate="+10%",
            volume="-5%",
            pitch="+2Hz",
        )
        assert config.voice == "en-GB-RyanNeural"
        assert config.rate == "+10%"

    def test_tts_config_defaults(self):
        """Test TTS config default values."""
        config = TTSConfig()
        assert config.voice == "en-US-AriaNeural"
        assert config.rate == "+0%"
        assert config.volume == "+0%"
        assert config.pitch == "+0Hz"


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    def test_create_conversation_turn(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            role="user",
            content="Hello there!",
            audio_duration=1.5,
        )
        assert turn.role == "user"
        assert turn.content == "Hello there!"
        assert turn.audio_duration == 1.5
        assert isinstance(turn.timestamp, datetime)

    def test_conversation_turn_roles(self):
        """Test valid conversation roles."""
        for role in ["user", "assistant", "system"]:
            turn = ConversationTurn(role=role, content="Test")
            assert turn.role == role


class TestAudioChunk:
    """Tests for AudioChunk model."""

    def test_create_audio_chunk(self):
        """Test creating an audio chunk."""
        chunk = AudioChunk(
            data=b"\x00\x01\x02\x03",
            sample_rate=16000,
            channels=1,
        )
        assert chunk.data == b"\x00\x01\x02\x03"
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1

    def test_audio_chunk_defaults(self):
        """Test audio chunk default values."""
        chunk = AudioChunk(data=b"\x00")
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1


class TestTranscriptionResult:
    """Tests for TranscriptionResult model."""

    def test_create_transcription_result(self):
        """Test creating a transcription result."""
        result = TranscriptionResult(
            text="Hello, world!",
            confidence=0.98,
            language="en",
            duration=1.2,
        )
        assert result.text == "Hello, world!"
        assert result.confidence == 0.98
        assert result.language == "en"
        assert result.duration == 1.2

    def test_transcription_result_defaults(self):
        """Test transcription result defaults."""
        result = TranscriptionResult(text="Test")
        assert result.confidence == 1.0
        assert result.language == "en"
        assert result.duration == 0.0


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        tool_call = ToolCall(
            name="get_weather",
            arguments={"city": "London", "units": "celsius"},
        )
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"city": "London", "units": "celsius"}

    def test_tool_call_empty_arguments(self):
        """Test tool call with no arguments."""
        tool_call = ToolCall(name="get_time")
        assert tool_call.arguments == {}


class TestToolResult:
    """Tests for ToolResult model."""

    def test_create_tool_result_success(self):
        """Test creating a successful tool result."""
        result = ToolResult(
            name="get_weather",
            result={"temperature": 20},
            success=True,
        )
        assert result.name == "get_weather"
        assert result.result == {"temperature": 20}
        assert result.success is True
        assert result.error is None

    def test_create_tool_result_failure(self):
        """Test creating a failed tool result."""
        result = ToolResult(
            name="get_weather",
            result=None,
            success=False,
            error="API unavailable",
        )
        assert result.success is False
        assert result.error == "API unavailable"


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_create_llm_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            text="Hello! How can I help?",
            tool_calls=[],
            finish_reason="stop",
        )
        assert response.text == "Hello! How can I help?"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    def test_llm_response_with_tool_calls(self, sample_llm_response_with_tool):
        """Test LLM response with tool calls."""
        assert len(sample_llm_response_with_tool.tool_calls) == 1
        assert sample_llm_response_with_tool.tool_calls[0].name == "get_weather"


class TestVoiceAgentConfig:
    """Tests for VoiceAgentConfig model."""

    def test_create_voice_agent_config(self):
        """Test creating voice agent config."""
        config = VoiceAgentConfig(
            model_name="gemini-1.5-pro",
            system_prompt="Be concise.",
            max_conversation_turns=10,
            interrupt_enabled=False,
        )
        assert config.model_name == "gemini-1.5-pro"
        assert config.system_prompt == "Be concise."
        assert config.max_conversation_turns == 10
        assert config.interrupt_enabled is False

    def test_voice_agent_config_defaults(self):
        """Test voice agent config defaults."""
        config = VoiceAgentConfig()
        assert config.model_name == "gemini-2.5-flash"
        assert "helpful" in config.system_prompt.lower()
        assert config.max_conversation_turns == 20
        assert config.interrupt_enabled is True

    def test_voice_agent_config_nested(self, sample_vad_config, sample_tts_config):
        """Test voice agent config with nested configs."""
        config = VoiceAgentConfig(
            vad=sample_vad_config,
            tts=sample_tts_config,
        )
        assert config.vad.threshold == 0.5
        assert config.tts.voice == "en-US-AriaNeural"


class TestRoomConfig:
    """Tests for RoomConfig model."""

    def test_create_room_config(self):
        """Test creating room config."""
        config = RoomConfig(
            url="ws://localhost:7880",
            api_key="devkey",
            api_secret="secret",
            room_name="test-room",
        )
        assert config.url == "ws://localhost:7880"
        assert config.api_key == "devkey"
        assert config.room_name == "test-room"


class TestAgentMetrics:
    """Tests for AgentMetrics model."""

    def test_create_agent_metrics(self):
        """Test creating agent metrics."""
        metrics = AgentMetrics(
            total_turns=5,
            avg_response_time=200.0,
            avg_transcription_time=50.0,
            avg_tts_time=100.0,
            total_audio_duration=15.0,
            error_count=1,
        )
        assert metrics.total_turns == 5
        assert metrics.avg_response_time == 200.0
        assert metrics.error_count == 1

    def test_agent_metrics_defaults(self):
        """Test agent metrics defaults."""
        metrics = AgentMetrics()
        assert metrics.total_turns == 0
        assert metrics.avg_response_time == 0.0
        assert metrics.error_count == 0
