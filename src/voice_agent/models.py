"""Pydantic models for the Real-Time Voice Agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Voice agent states."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    role: Literal["user", "assistant", "system"] = Field(description="Speaker role")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    audio_duration: float | None = Field(default=None, description="Duration in seconds")


class VADConfig(BaseModel):
    """Voice Activity Detection configuration."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection threshold")
    min_speech_duration: float = Field(default=0.25, description="Minimum speech duration (s)")
    min_silence_duration: float = Field(default=0.5, description="Minimum silence duration (s)")
    sample_rate: int = Field(default=16000, description="Audio sample rate")


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""

    voice: str = Field(default="en-US-AriaNeural", description="TTS voice name")
    rate: str = Field(default="+0%", description="Speech rate adjustment")
    volume: str = Field(default="+0%", description="Volume adjustment")
    pitch: str = Field(default="+0Hz", description="Pitch adjustment")


class AudioChunk(BaseModel):
    """An audio chunk for processing."""

    data: bytes = Field(description="Raw audio data")
    sample_rate: int = Field(default=16000, description="Sample rate")
    channels: int = Field(default=1, description="Number of channels")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}


class TranscriptionResult(BaseModel):
    """Speech-to-text transcription result."""

    text: str = Field(description="Transcribed text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    language: str = Field(default="en", description="Detected language")
    duration: float = Field(default=0.0, description="Audio duration in seconds")


class LLMResponse(BaseModel):
    """Response from the LLM."""

    text: str = Field(description="Response text")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls to execute")
    finish_reason: str | None = Field(default=None, description="Why generation stopped")


class ToolCall(BaseModel):
    """A tool call from the LLM."""

    name: str = Field(description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolResult(BaseModel):
    """Result from executing a tool."""

    name: str = Field(description="Tool name")
    result: Any = Field(description="Tool execution result")
    success: bool = Field(default=True, description="Whether execution succeeded")
    error: str | None = Field(default=None, description="Error message if failed")


# Fix forward reference
LLMResponse.model_rebuild()


class VoiceAgentConfig(BaseModel):
    """Configuration for the voice agent."""

    model_name: str = Field(default="gemini-2.5-flash", description="LLM model name")
    vad: VADConfig = Field(default_factory=VADConfig, description="VAD configuration")
    tts: TTSConfig = Field(default_factory=TTSConfig, description="TTS configuration")
    system_prompt: str = Field(
        default="You are a helpful voice assistant. Keep responses concise and conversational.",
        description="System prompt for the LLM",
    )
    max_conversation_turns: int = Field(default=20, description="Max turns to keep in context")
    interrupt_enabled: bool = Field(default=True, description="Allow user interruptions")


class RoomConfig(BaseModel):
    """LiveKit room configuration."""

    url: str = Field(description="LiveKit server URL")
    api_key: str = Field(description="API key")
    api_secret: str = Field(description="API secret")
    room_name: str = Field(default="voice-agent-room", description="Room name")


class AgentMetrics(BaseModel):
    """Metrics for monitoring agent performance."""

    total_turns: int = Field(default=0, description="Total conversation turns")
    avg_response_time: float = Field(default=0.0, description="Average response time (ms)")
    avg_transcription_time: float = Field(default=0.0, description="Average STT time (ms)")
    avg_tts_time: float = Field(default=0.0, description="Average TTS time (ms)")
    total_audio_duration: float = Field(default=0.0, description="Total audio processed (s)")
    error_count: int = Field(default=0, description="Number of errors")
