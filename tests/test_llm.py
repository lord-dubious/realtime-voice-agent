"""Tests for LLM integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_agent.llm import GeminiConfigurationError, GeminiDependencyError, GeminiLLM, create_llm
from voice_agent.models import ConversationTurn, LLMResponse, ToolCall, VoiceAgentConfig


class TestGeminiLLM:
    """Tests for GeminiLLM class."""

    def test_init_default_config(self):
        """Test LLM initialization with default config."""
        llm = GeminiLLM()
        assert llm.config.model_name == "gemini-2.5-flash"
        assert llm._model is None
        assert llm._tools == {}

    def test_init_custom_config(self, sample_agent_config):
        """Test LLM initialization with custom config."""
        llm = GeminiLLM(config=sample_agent_config)
        assert llm.config == sample_agent_config

    def test_register_tool(self):
        """Test registering a tool."""
        llm = GeminiLLM()

        def my_tool(arg: str) -> str:
            return f"Result: {arg}"

        llm.register_tool("my_tool", my_tool, "A test tool")

        assert "my_tool" in llm._tools
        assert llm._tools["my_tool"]["func"] == my_tool
        assert llm._tools["my_tool"]["description"] == "A test tool"

    @pytest.mark.asyncio
    async def test_generate_mock_response(self):
        """Test generating with mock model."""
        llm = GeminiLLM.create_mock()

        response = await llm.generate("hello")

        assert isinstance(response, LLMResponse)
        assert "hello" in response.text.lower() or "help" in response.text.lower()
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_history(self):
        """Test generating with conversation history."""
        llm = GeminiLLM.create_mock()

        history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ]

        response = await llm.generate("how are you", history)

        assert isinstance(response, LLMResponse)
        assert "well" in response.text.lower() or "how" in response.text.lower()

    @pytest.mark.asyncio
    async def test_generate_with_gemini(self, mock_gemini_model):
        """Test generating with Gemini model."""
        llm = GeminiLLM()
        llm._model = mock_gemini_model

        with patch("google.generativeai.GenerativeModel", return_value=mock_gemini_model):
            response = await llm.generate("Hello")

            assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_generate_handles_error_with_logging(self, caplog, capsys):
        """Test error handling in generate logs instead of printing."""
        llm = GeminiLLM()

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API error")
        llm._model = mock_model

        with caplog.at_level("ERROR", logger="voice_agent.llm"):
            response = await llm.generate("Test")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "LLM generation error" in caplog.text
        assert "error" in response.text.lower() or "sorry" in response.text.lower()
        assert response.finish_reason == "error"

    def test_get_model_fails_when_dependency_missing(self):
        """Test real Gemini mode fails clearly when the SDK is missing."""
        llm = GeminiLLM()

        with (
            patch(
                "voice_agent.llm.import_module",
                side_effect=ImportError("missing google-generativeai"),
            ),
            pytest.raises(GeminiDependencyError, match="google-generativeai is required"),
        ):
            llm._get_model()

    def test_get_model_fails_when_api_key_missing(self, monkeypatch):
        """Test real Gemini mode fails clearly when GEMINI_API_KEY is missing."""
        llm = GeminiLLM()
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY is required"):
            llm._get_model()

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming generation."""
        llm = GeminiLLM.create_mock()

        chunks = []
        async for chunk in llm.generate_stream("hello"):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    def test_parse_tool_calls(self):
        """Test parsing tool calls from text."""
        llm = GeminiLLM()

        text = "Let me check that. [TOOL:get_weather(city=London, units=celsius)]"
        tool_calls = llm._parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"city": "London", "units": "celsius"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        llm = GeminiLLM()

        text = "[TOOL:get_weather(city=London)] and also [TOOL:get_time()]"
        tool_calls = llm._parse_tool_calls(text)

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[1].name == "get_time"

    def test_parse_no_tool_calls(self):
        """Test parsing text with no tool calls."""
        llm = GeminiLLM()

        text = "Just a regular response without any tools."
        tool_calls = llm._parse_tool_calls(text)

        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        llm = GeminiLLM()

        def test_func(value: str) -> str:
            return f"Got: {value}"

        llm.register_tool("test", test_func, "Test tool")

        tool_call = ToolCall(name="test", arguments={"value": "hello"})
        result = await llm.execute_tool(tool_call)

        assert result == "Got: hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        llm = GeminiLLM()

        tool_call = ToolCall(name="unknown", arguments={})
        result = await llm.execute_tool(tool_call)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """Test tool execution error handling."""
        llm = GeminiLLM()

        def broken_func():
            raise ValueError("Tool failed")

        llm.register_tool("broken", broken_func, "Broken tool")

        tool_call = ToolCall(name="broken", arguments={})
        result = await llm.execute_tool(tool_call)

        assert "error" in result

    def test_build_messages(self):
        """Test building messages for the LLM."""
        llm = GeminiLLM()

        history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ]

        messages = llm._build_messages("How are you?", history)

        # Should have system prompt + history + current input
        assert len(messages) >= 5  # system (2) + history (2) + current (1)
        assert messages[-1]["parts"] == ["How are you?"]

    def test_build_messages_truncates_history(self):
        """Test that history is truncated to max turns."""
        config = VoiceAgentConfig(max_conversation_turns=2)
        llm = GeminiLLM(config=config)

        history = [ConversationTurn(role="user", content=f"Message {i}") for i in range(10)]

        messages = llm._build_messages("Current", history)

        # Should only include last 2 turns from history
        history_messages = [m for m in messages if m["parts"][0].startswith("Message")]
        assert len(history_messages) == 2


class TestCreateLLM:
    """Tests for create_llm factory function."""

    def test_create_llm_default(self):
        """Test creating LLM with defaults."""
        llm = create_llm()
        assert isinstance(llm, GeminiLLM)

    def test_create_llm_with_config(self, sample_agent_config):
        """Test creating LLM with custom config."""
        llm = create_llm(sample_agent_config)
        assert llm.config == sample_agent_config

    def test_create_llm_mock(self):
        """Test creating an explicitly mocked LLM through the factory."""
        llm = create_llm(mock=True)
        assert isinstance(llm, GeminiLLM)
        assert llm._mock_mode is True


class TestMockResponses:
    """Tests for mock response generation."""

    @pytest.mark.asyncio
    async def test_mock_greeting(self):
        """Test mock response to greeting."""
        llm = GeminiLLM.create_mock()

        response = await llm.generate("hello")
        assert "hello" in response.text.lower() or "help" in response.text.lower()

    @pytest.mark.asyncio
    async def test_mock_goodbye(self):
        """Test mock response to goodbye."""
        llm = GeminiLLM.create_mock()

        response = await llm.generate("goodbye")
        assert "goodbye" in response.text.lower() or "great day" in response.text.lower()

    @pytest.mark.asyncio
    async def test_mock_unknown(self):
        """Test mock response to unknown input."""
        llm = GeminiLLM.create_mock()

        response = await llm.generate("xyzzy12345")
        assert len(response.text) > 0
