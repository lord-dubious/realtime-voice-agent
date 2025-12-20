"""LLM integration using Google Gemini.

This module provides the AI brain for the voice agent using
Gemini for fast, intelligent responses.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from voice_agent.models import ConversationTurn, LLMResponse, ToolCall, VoiceAgentConfig

load_dotenv()


class GeminiLLM:
    """Gemini LLM wrapper for voice agent intelligence.

    Uses Gemini Flash for low-latency responses suitable for
    real-time voice interactions.
    """

    def __init__(self, config: VoiceAgentConfig | None = None):
        """Initialize the LLM.

        Args:
            config: Voice agent configuration.
        """
        self.config = config or VoiceAgentConfig()
        self._model = None
        self._tools: dict[str, callable] = {}

    def _get_model(self):
        """Lazy load the Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai

                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not set")

                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(self.config.model_name)

            except ImportError:
                print("google-generativeai not installed, using mock")
                self._model = "mock"

        return self._model

    def register_tool(self, name: str, func: callable, description: str = "") -> None:
        """Register a tool for the LLM to use.

        Args:
            name: Tool name.
            func: Tool function.
            description: Tool description.
        """
        self._tools[name] = {"func": func, "description": description}

    async def generate(
        self,
        user_input: str,
        conversation_history: list[ConversationTurn] | None = None,
    ) -> LLMResponse:
        """Generate a response to user input.

        Args:
            user_input: User's message.
            conversation_history: Previous conversation turns.

        Returns:
            LLM response with text and optional tool calls.
        """
        model = self._get_model()

        if model == "mock":
            return self._mock_response(user_input)

        try:
            import google.generativeai as genai

            # Build conversation context
            messages = self._build_messages(user_input, conversation_history or [])

            # Generate response
            response = model.generate_content(messages)

            # Parse response
            text = response.text if response.text else ""
            tool_calls = self._parse_tool_calls(text)

            return LLMResponse(
                text=text,
                tool_calls=tool_calls,
                finish_reason=str(response.candidates[0].finish_reason)
                if response.candidates
                else None,
            )

        except Exception as e:
            print(f"LLM generation error: {e}")
            return LLMResponse(
                text="I'm sorry, I encountered an error. Could you please repeat that?",
                tool_calls=[],
                finish_reason="error",
            )

    async def generate_stream(
        self,
        user_input: str,
        conversation_history: list[ConversationTurn] | None = None,
    ):
        """Stream a response to user input.

        Args:
            user_input: User's message.
            conversation_history: Previous conversation turns.

        Yields:
            Text chunks.
        """
        model = self._get_model()

        if model == "mock":
            for word in self._mock_response(user_input).text.split():
                yield word + " "
            return

        try:
            messages = self._build_messages(user_input, conversation_history or [])
            response = model.generate_content(messages, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            print(f"LLM streaming error: {e}")
            yield "I'm sorry, I encountered an error."

    def _build_messages(
        self,
        user_input: str,
        history: list[ConversationTurn],
    ) -> list[dict]:
        """Build message list for the LLM.

        Args:
            user_input: Current user input.
            history: Conversation history.

        Returns:
            List of message dictionaries.
        """
        messages = []

        # Add system prompt
        messages.append({"role": "user", "parts": [self.config.system_prompt]})
        messages.append({"role": "model", "parts": ["Understood. I'm ready to help."]})

        # Add history (limited to max turns)
        recent_history = history[-self.config.max_conversation_turns :]
        for turn in recent_history:
            role = "user" if turn.role == "user" else "model"
            messages.append({"role": role, "parts": [turn.content]})

        # Add current input
        messages.append({"role": "user", "parts": [user_input]})

        return messages

    def _parse_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse tool calls from response text.

        This is a simple implementation - production would use
        structured function calling.

        Args:
            text: Response text.

        Returns:
            List of tool calls.
        """
        # Simple pattern matching for tool calls
        # Format: [TOOL:name(arg1=val1, arg2=val2)]
        tool_calls = []
        import re

        pattern = r"\[TOOL:(\w+)\((.*?)\)\]"
        matches = re.findall(pattern, text)

        for name, args_str in matches:
            args = {}
            if args_str:
                for pair in args_str.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        args[key.strip()] = value.strip().strip("\"'")

            tool_calls.append(ToolCall(name=name, arguments=args))

        return tool_calls

    async def execute_tool(self, tool_call: ToolCall) -> Any:
        """Execute a tool call.

        Args:
            tool_call: Tool call to execute.

        Returns:
            Tool execution result.
        """
        if tool_call.name not in self._tools:
            return {"error": f"Unknown tool: {tool_call.name}"}

        tool = self._tools[tool_call.name]
        try:
            result = tool["func"](**tool_call.arguments)
            return result
        except Exception as e:
            return {"error": str(e)}

    def _mock_response(self, user_input: str) -> LLMResponse:
        """Generate a mock response for testing.

        Args:
            user_input: User input.

        Returns:
            Mock LLM response.
        """
        responses = {
            "hello": "Hello! How can I help you today?",
            "how are you": "I'm doing well, thank you for asking! How can I assist you?",
            "what time is it": "I don't have access to the current time, but I'd be happy to help with something else!",
            "goodbye": "Goodbye! Have a great day!",
        }

        user_lower = user_input.lower()
        for key, response in responses.items():
            if key in user_lower:
                return LLMResponse(text=response, tool_calls=[], finish_reason="stop")

        return LLMResponse(
            text=f"I heard you say: '{user_input}'. How can I help you with that?",
            tool_calls=[],
            finish_reason="stop",
        )


def create_llm(config: VoiceAgentConfig | None = None) -> GeminiLLM:
    """Create a GeminiLLM instance.

    Args:
        config: Optional voice agent configuration.

    Returns:
        GeminiLLM instance.
    """
    return GeminiLLM(config)
