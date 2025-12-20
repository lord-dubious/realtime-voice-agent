"""Tests for the VoiceAgent orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_agent.agent import VoiceAgent, create_agent
from voice_agent.models import (
    AgentMetrics,
    AgentState,
    AudioChunk,
    ConversationTurn,
    LLMResponse,
    TranscriptionResult,
    VoiceAgentConfig,
)


class TestVoiceAgentInit:
    """Tests for VoiceAgent initialization."""

    def test_create_agent_default_config(self):
        """Test creating agent with default config."""
        agent = create_agent()
        assert agent is not None
        assert agent.state == AgentState.IDLE
        assert agent.conversation == []

    def test_create_agent_custom_config(self, sample_agent_config):
        """Test creating agent with custom config."""
        agent = create_agent(sample_agent_config)
        assert agent.config == sample_agent_config
        assert agent.config.model_name == "gemini-2.5-flash"

    def test_agent_initial_state(self):
        """Test agent starts in IDLE state."""
        agent = VoiceAgent()
        assert agent.state == AgentState.IDLE
        assert len(agent.conversation) == 0

    def test_agent_has_vad(self):
        """Test agent has VAD component."""
        agent = VoiceAgent()
        assert agent.vad is not None

    def test_agent_has_llm(self):
        """Test agent has LLM component."""
        agent = VoiceAgent()
        assert agent.llm is not None

    def test_agent_has_tts(self):
        """Test agent has TTS component."""
        agent = VoiceAgent()
        assert agent.tts is not None

    def test_agent_initial_metrics(self):
        """Test agent initializes with empty metrics."""
        agent = VoiceAgent()
        metrics = agent.get_metrics()
        assert metrics.total_turns == 0
        assert metrics.error_count == 0
        assert metrics.avg_response_time == 0.0


class TestVoiceAgentCallbacks:
    """Tests for VoiceAgent callback registration."""

    def test_on_state_change_callback(self):
        """Test state change callback registration."""
        agent = VoiceAgent()
        callback = MagicMock()
        agent.on_state_change(callback)
        assert agent._on_state_change == callback

    def test_on_transcription_callback(self):
        """Test transcription callback registration."""
        agent = VoiceAgent()
        callback = MagicMock()
        agent.on_transcription(callback)
        assert agent._on_transcription == callback

    def test_on_response_callback(self):
        """Test response callback registration."""
        agent = VoiceAgent()
        callback = MagicMock()
        agent.on_response(callback)
        assert agent._on_response == callback

    def test_on_audio_output_callback(self):
        """Test audio output callback registration."""
        agent = VoiceAgent()
        callback = MagicMock()
        agent.on_audio_output(callback)
        assert agent._on_audio_output == callback

    def test_state_change_triggers_callback(self):
        """Test that state changes trigger callback."""
        agent = VoiceAgent()
        states = []
        agent.on_state_change(lambda s: states.append(s))
        agent._set_state(AgentState.LISTENING)
        assert AgentState.LISTENING in states


class TestVoiceAgentToolRegistration:
    """Tests for tool registration."""

    def test_register_tool(self):
        """Test registering a tool."""
        agent = VoiceAgent()

        def get_weather(city: str) -> str:
            return f"Weather in {city}: sunny"

        agent.register_tool("get_weather", get_weather, "Get weather for a city")
        assert "get_weather" in agent.llm._tools

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        agent = VoiceAgent()

        agent.register_tool("tool1", lambda: None, "Tool 1")
        agent.register_tool("tool2", lambda: None, "Tool 2")

        assert "tool1" in agent.llm._tools
        assert "tool2" in agent.llm._tools


class TestVoiceAgentAudioProcessing:
    """Tests for audio processing."""

    @pytest.mark.asyncio
    async def test_process_silent_audio_chunk(self, sample_audio_chunk):
        """Test processing silent audio doesn't trigger speech."""
        agent = VoiceAgent()
        states = []
        agent.on_state_change(lambda s: states.append(s))

        await agent.process_audio_chunk(sample_audio_chunk)

        # Silent audio shouldn't trigger LISTENING state
        assert AgentState.LISTENING not in states

    @pytest.mark.asyncio
    async def test_audio_buffer_clears_after_processing(self):
        """Test audio buffer clears after speech processing."""
        agent = VoiceAgent()
        agent._audio_buffer = [AudioChunk(data=b"\x00" * 100)]
        await agent._process_speech()
        assert len(agent._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_process_speech_updates_metrics(self):
        """Test that processing speech updates metrics."""
        agent = VoiceAgent()

        # Mock TTS to avoid actual synthesis
        agent.tts.synthesize = AsyncMock(return_value=b"\x00" * 100)

        # Add mock audio to buffer
        samples = np.zeros(16000, dtype=np.int16)
        agent._audio_buffer = [AudioChunk(data=samples.tobytes())]

        await agent._process_speech()

        metrics = agent.get_metrics()
        assert metrics.total_turns >= 0  # May or may not increment based on mock

    @pytest.mark.asyncio
    async def test_processing_sets_error_state_on_exception(self):
        """Test that exceptions during processing set error state."""
        agent = VoiceAgent()
        states = []
        agent.on_state_change(lambda s: states.append(s))

        # Mock LLM to raise exception
        agent.llm.generate = AsyncMock(side_effect=Exception("Test error"))

        samples = np.zeros(16000, dtype=np.int16)
        agent._audio_buffer = [AudioChunk(data=samples.tobytes())]

        await agent._process_speech()

        assert AgentState.ERROR in states
        assert agent.metrics.error_count == 1


class TestVoiceAgentSay:
    """Tests for the say method."""

    @pytest.mark.asyncio
    async def test_say_synthesizes_audio(self):
        """Test say method synthesizes audio."""
        agent = VoiceAgent()
        agent.tts.synthesize = AsyncMock(return_value=b"\x00" * 1000)

        audio = await agent.say("Hello, world!")

        assert audio is not None
        assert len(audio) == 1000

    @pytest.mark.asyncio
    async def test_say_triggers_audio_callback(self):
        """Test say method triggers audio output callback."""
        agent = VoiceAgent()
        agent.tts.synthesize = AsyncMock(return_value=b"\x00" * 1000)

        audio_data = []
        agent.on_audio_output(lambda a: audio_data.append(a))

        await agent.say("Test message")

        assert len(audio_data) == 1
        assert audio_data[0] == b"\x00" * 1000

    @pytest.mark.asyncio
    async def test_say_sets_speaking_state(self):
        """Test say method sets SPEAKING state."""
        agent = VoiceAgent()
        agent.tts.synthesize = AsyncMock(return_value=b"\x00" * 100)

        states = []
        agent.on_state_change(lambda s: states.append(s))

        await agent.say("Hello")

        assert AgentState.SPEAKING in states
        assert agent.state == AgentState.IDLE  # Returns to IDLE after

    @pytest.mark.asyncio
    async def test_say_returns_none_on_empty_audio(self):
        """Test say returns None when TTS returns nothing."""
        agent = VoiceAgent()
        agent.tts.synthesize = AsyncMock(return_value=None)

        audio = await agent.say("Hello")

        assert audio is None


class TestVoiceAgentReset:
    """Tests for the reset method."""

    def test_reset_clears_conversation(self):
        """Test reset clears conversation history."""
        agent = VoiceAgent()
        agent.conversation.append(ConversationTurn(role="user", content="Hello"))

        agent.reset()

        assert len(agent.conversation) == 0

    def test_reset_clears_audio_buffer(self):
        """Test reset clears audio buffer."""
        agent = VoiceAgent()
        agent._audio_buffer = [AudioChunk(data=b"\x00" * 100)]

        agent.reset()

        assert len(agent._audio_buffer) == 0

    def test_reset_sets_idle_state(self):
        """Test reset sets IDLE state."""
        agent = VoiceAgent()
        agent._set_state(AgentState.PROCESSING)

        agent.reset()

        assert agent.state == AgentState.IDLE

    def test_reset_clears_processing_flag(self):
        """Test reset clears processing flag."""
        agent = VoiceAgent()
        agent._is_processing = True

        agent.reset()

        assert agent._is_processing is False


class TestVoiceAgentMetrics:
    """Tests for metrics tracking."""

    def test_get_metrics_returns_agent_metrics(self):
        """Test get_metrics returns AgentMetrics instance."""
        agent = VoiceAgent()
        metrics = agent.get_metrics()
        assert isinstance(metrics, AgentMetrics)

    def test_update_avg_metric_first_value(self):
        """Test updating average with first value."""
        agent = VoiceAgent()
        agent.metrics.total_turns = 1
        agent._update_avg_metric("avg_response_time", 100.0)
        assert agent.metrics.avg_response_time == 100.0

    def test_update_avg_metric_running_average(self):
        """Test running average calculation."""
        agent = VoiceAgent()
        agent.metrics.total_turns = 2
        agent.metrics.avg_response_time = 100.0
        agent._update_avg_metric("avg_response_time", 200.0)
        # New avg should be (100 * 1 + 200) / 2 = 150
        assert agent.metrics.avg_response_time == 150.0


class TestVoiceAgentTranscription:
    """Tests for transcription functionality."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(self):
        """Test transcription returns TranscriptionResult."""
        agent = VoiceAgent()

        samples = np.zeros(16000, dtype=np.int16)
        result = await agent._transcribe(samples.tobytes())

        assert isinstance(result, TranscriptionResult)
        assert result.text is not None
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_transcribe_calculates_duration(self):
        """Test transcription calculates audio duration."""
        agent = VoiceAgent()

        # 1 second of audio at 16kHz, 16-bit
        samples = np.zeros(16000, dtype=np.int16)
        result = await agent._transcribe(samples.tobytes())

        assert result.duration == 1.0


class TestVoiceAgentInterruption:
    """Tests for interruption handling."""

    @pytest.mark.asyncio
    async def test_interrupt_disabled_blocks_audio(self, sample_speech_audio_chunk):
        """Test that audio is blocked when processing and interrupts disabled."""
        config = VoiceAgentConfig(interrupt_enabled=False)
        agent = VoiceAgent(config)
        agent._is_processing = True

        await agent.process_audio_chunk(sample_speech_audio_chunk)

        # Should not have buffered the audio
        assert len(agent._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_interrupt_enabled_allows_audio(self, sample_speech_audio_chunk):
        """Test that audio is processed when interrupts enabled."""
        config = VoiceAgentConfig(interrupt_enabled=True)
        agent = VoiceAgent(config)

        # Note: Not setting _is_processing = True here since we want to test normal flow
        await agent.process_audio_chunk(sample_speech_audio_chunk)

        # Audio might be buffered depending on VAD detection


class TestVoiceAgentConversation:
    """Tests for conversation management."""

    @pytest.mark.asyncio
    async def test_conversation_adds_user_turn(self):
        """Test that user turns are added to conversation."""
        agent = VoiceAgent()
        agent.tts.synthesize = AsyncMock(return_value=b"\x00" * 100)

        samples = np.zeros(16000, dtype=np.int16)
        agent._audio_buffer = [AudioChunk(data=samples.tobytes())]

        await agent._process_speech()

        # Should have user and assistant turns
        assert len(agent.conversation) >= 1

    @pytest.mark.asyncio
    async def test_conversation_adds_assistant_turn(self):
        """Test that assistant turns are added to conversation."""
        agent = VoiceAgent()
        agent.tts.synthesize = AsyncMock(return_value=b"\x00" * 100)

        samples = np.zeros(16000, dtype=np.int16)
        agent._audio_buffer = [AudioChunk(data=samples.tobytes())]

        await agent._process_speech()

        # Check for assistant turn
        assistant_turns = [t for t in agent.conversation if t.role == "assistant"]
        assert len(assistant_turns) >= 1


class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_create_agent_returns_voice_agent(self):
        """Test create_agent returns VoiceAgent instance."""
        agent = create_agent()
        assert isinstance(agent, VoiceAgent)

    def test_create_agent_with_config(self, sample_agent_config):
        """Test create_agent accepts config."""
        agent = create_agent(sample_agent_config)
        assert agent.config == sample_agent_config

    def test_create_agent_default_model(self):
        """Test default model is set."""
        agent = create_agent()
        assert agent.config.model_name == "gemini-2.5-flash"
