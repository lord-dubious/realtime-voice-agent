"""Real-Time Voice Agent - WebRTC voice interaction demo with Gemini AI.

A developer-focused voice agent using:
- LiveKit for WebRTC transport
- Silero VAD for voice activity detection
- Gemini for AI reasoning, with explicit dependency/config failures
- Edge TTS for speech synthesis

Transcription is currently a demo placeholder; integrate a real STT backend
before using this package with real users.
"""

from voice_agent.agent import VoiceAgent, create_agent
from voice_agent.llm import (
    GeminiConfigurationError,
    GeminiDependencyError,
    GeminiGenerationError,
    GeminiLLM,
    create_llm,
)
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
from voice_agent.tts import TextToSpeech, create_tts
from voice_agent.vad import VoiceActivityDetector, create_vad

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "VoiceAgent",
    "VoiceActivityDetector",
    "GeminiLLM",
    "GeminiConfigurationError",
    "GeminiDependencyError",
    "GeminiGenerationError",
    "TextToSpeech",
    # Factory functions
    "create_agent",
    "create_vad",
    "create_llm",
    "create_tts",
    # Models
    "AgentState",
    "AgentMetrics",
    "ConversationTurn",
    "VADConfig",
    "TTSConfig",
    "VoiceAgentConfig",
    "RoomConfig",
    "AudioChunk",
    "TranscriptionResult",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    # Version
    "__version__",
]
