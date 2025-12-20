"""Voice Agent core orchestration.

This module provides the main VoiceAgent class that coordinates
VAD, STT, LLM, and TTS components for real-time voice interactions.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Callable

from dotenv import load_dotenv

from voice_agent.llm import GeminiLLM, create_llm
from voice_agent.models import (
    AgentMetrics,
    AgentState,
    AudioChunk,
    ConversationTurn,
    TranscriptionResult,
    VoiceAgentConfig,
)
from voice_agent.tts import TextToSpeech, create_tts
from voice_agent.vad import VoiceActivityDetector, create_vad

load_dotenv()


class VoiceAgent:
    """Real-time voice agent with VAD, LLM, and TTS.

    Orchestrates the complete voice interaction pipeline:
    1. VAD detects when user is speaking
    2. Audio is transcribed to text
    3. LLM generates a response
    4. TTS synthesizes speech output
    """

    def __init__(self, config: VoiceAgentConfig | None = None):
        """Initialize the voice agent.

        Args:
            config: Agent configuration.
        """
        self.config = config or VoiceAgentConfig()
        self.state = AgentState.IDLE
        self.conversation: list[ConversationTurn] = []
        self.metrics = AgentMetrics()

        # Initialize components
        self.vad = create_vad(self.config.vad)
        self.llm = create_llm(self.config)
        self.tts = create_tts(self.config.tts)

        # Callbacks
        self._on_state_change: Callable[[AgentState], None] | None = None
        self._on_transcription: Callable[[str], None] | None = None
        self._on_response: Callable[[str], None] | None = None
        self._on_audio_output: Callable[[bytes], None] | None = None

        # Audio buffer for collecting speech
        self._audio_buffer: list[AudioChunk] = []
        self._is_processing = False

    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool for the LLM to use.

        Args:
            name: Tool name.
            func: Tool function.
            description: Tool description.
        """
        self.llm.register_tool(name, func, description)

    def on_state_change(self, callback: Callable[[AgentState], None]) -> None:
        """Register state change callback.

        Args:
            callback: Function to call on state changes.
        """
        self._on_state_change = callback

    def on_transcription(self, callback: Callable[[str], None]) -> None:
        """Register transcription callback.

        Args:
            callback: Function to call when transcription is ready.
        """
        self._on_transcription = callback

    def on_response(self, callback: Callable[[str], None]) -> None:
        """Register response callback.

        Args:
            callback: Function to call when LLM response is ready.
        """
        self._on_response = callback

    def on_audio_output(self, callback: Callable[[bytes], None]) -> None:
        """Register audio output callback.

        Args:
            callback: Function to call when TTS audio is ready.
        """
        self._on_audio_output = callback

    def _set_state(self, state: AgentState) -> None:
        """Update agent state and notify callbacks.

        Args:
            state: New state.
        """
        self.state = state
        if self._on_state_change:
            self._on_state_change(state)

    async def process_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process an incoming audio chunk.

        Args:
            chunk: Audio chunk to process.
        """
        if self._is_processing and not self.config.interrupt_enabled:
            return

        # Convert to numpy for VAD
        import numpy as np

        audio = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Check for speech
        has_speech = self.vad.detect_speech(audio)

        if has_speech:
            if self.state != AgentState.LISTENING:
                self._set_state(AgentState.LISTENING)

            self._audio_buffer.append(chunk)
            self.metrics.total_audio_duration += len(chunk.data) / (2 * self.config.vad.sample_rate)

        elif self.state == AgentState.LISTENING and len(self._audio_buffer) > 0:
            # End of speech detected
            await self._process_speech()

    async def _process_speech(self) -> None:
        """Process collected speech buffer."""
        if not self._audio_buffer:
            return

        self._is_processing = True
        self._set_state(AgentState.PROCESSING)

        try:
            # Combine audio chunks
            audio_data = b"".join(chunk.data for chunk in self._audio_buffer)

            # Transcribe (using mock for now - integrate Whisper in production)
            start_time = time.time()
            transcription = await self._transcribe(audio_data)
            transcription_time = (time.time() - start_time) * 1000

            if transcription.text:
                if self._on_transcription:
                    self._on_transcription(transcription.text)

                # Add to conversation
                self.conversation.append(
                    ConversationTurn(
                        role="user",
                        content=transcription.text,
                        audio_duration=transcription.duration,
                    )
                )

                # Generate response
                start_time = time.time()
                response = await self.llm.generate(transcription.text, self.conversation)
                response_time = (time.time() - start_time) * 1000

                if response.text:
                    if self._on_response:
                        self._on_response(response.text)

                    # Add to conversation
                    self.conversation.append(
                        ConversationTurn(role="assistant", content=response.text)
                    )

                    # Execute any tool calls
                    for tool_call in response.tool_calls:
                        result = await self.llm.execute_tool(tool_call)
                        # Could inject result back into response

                    # Synthesize speech
                    self._set_state(AgentState.SPEAKING)
                    start_time = time.time()
                    audio = await self.tts.synthesize(response.text)
                    tts_time = (time.time() - start_time) * 1000

                    if audio and self._on_audio_output:
                        self._on_audio_output(audio)

                    # Update metrics
                    self.metrics.total_turns += 1
                    self._update_avg_metric("avg_transcription_time", transcription_time)
                    self._update_avg_metric("avg_response_time", response_time)
                    self._update_avg_metric("avg_tts_time", tts_time)

        except Exception as e:
            print(f"Speech processing error: {e}")
            self.metrics.error_count += 1
            self._set_state(AgentState.ERROR)

        finally:
            self._audio_buffer.clear()
            self._is_processing = False
            self._set_state(AgentState.IDLE)

    async def _transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe audio to text.

        This is a placeholder - integrate Faster-Whisper or
        Gemini's audio API in production.

        Args:
            audio_data: Raw audio bytes.

        Returns:
            Transcription result.
        """
        # Mock transcription for demonstration
        # In production, use faster-whisper or Gemini audio API
        duration = len(audio_data) / (2 * self.config.vad.sample_rate)

        # For now, return a placeholder
        # Real implementation would use:
        # - faster-whisper for local transcription
        # - Gemini's audio capabilities
        # - Google Cloud Speech-to-Text

        return TranscriptionResult(
            text="Hello, this is a test transcription.",
            confidence=0.95,
            language="en",
            duration=duration,
        )

    def _update_avg_metric(self, name: str, value: float) -> None:
        """Update a running average metric.

        Args:
            name: Metric name.
            value: New value.
        """
        current = getattr(self.metrics, name)
        count = self.metrics.total_turns
        if count > 0:
            new_avg = (current * (count - 1) + value) / count
            setattr(self.metrics, name, new_avg)
        else:
            setattr(self.metrics, name, value)

    async def say(self, text: str) -> bytes | None:
        """Speak text without waiting for user input.

        Args:
            text: Text to speak.

        Returns:
            Audio data if successful.
        """
        self._set_state(AgentState.SPEAKING)
        try:
            audio = await self.tts.synthesize(text)
            if audio and self._on_audio_output:
                self._on_audio_output(audio)
            return audio
        finally:
            self._set_state(AgentState.IDLE)

    def reset(self) -> None:
        """Reset the agent state."""
        self.conversation.clear()
        self._audio_buffer.clear()
        self.vad.reset()
        self._is_processing = False
        self._set_state(AgentState.IDLE)

    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics.

        Returns:
            Agent metrics.
        """
        return self.metrics


def create_agent(config: VoiceAgentConfig | None = None) -> VoiceAgent:
    """Create a VoiceAgent instance.

    Args:
        config: Optional agent configuration.

    Returns:
        VoiceAgent instance.
    """
    return VoiceAgent(config)
