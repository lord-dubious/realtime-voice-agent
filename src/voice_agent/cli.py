"""CLI for Real-Time Voice Agent.

A Typer-based command-line interface for running the voice agent.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voice_agent.agent import create_agent
from voice_agent.models import AgentState, VoiceAgentConfig

app = typer.Typer(
    name="voice-agent",
    help="Real-time voice assistant demo with LiveKit, Silero VAD, Edge TTS, Gemini, and explicit mock boundaries",
    no_args_is_help=True,
)
console = Console()


def check_environment() -> bool:
    """Check if required environment variables are set.

    Returns:
        True if environment is properly configured.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] GEMINI_API_KEY environment variable not set.",
            style="bold",
        )
        console.print("Set it with: export GEMINI_API_KEY=your-api-key")
        return False
    return True


@app.command()
def run(
    model: str = typer.Option(
        "gemini-2.5-flash",
        "--model",
        "-m",
        help="Gemini model to use",
    ),
    voice: str = typer.Option(
        "en-US-AriaNeural",
        "--voice",
        "-v",
        help="TTS voice to use",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="VAD threshold (0.0-1.0)",
    ),
) -> None:
    """Run the voice agent in interactive mode."""
    if not check_environment():
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold blue]Real-Time Voice Agent[/bold blue]\n\nPress Ctrl+C to stop",
            border_style="blue",
        )
    )

    config = VoiceAgentConfig(
        model_name=model,
    )
    config.tts.voice = voice
    config.vad.threshold = threshold

    agent = create_agent(config)

    # Register callbacks
    def on_state_change(state: AgentState):
        status = {
            AgentState.IDLE: "[dim]Idle[/dim]",
            AgentState.LISTENING: "[green]Listening...[/green]",
            AgentState.PROCESSING: "[yellow]Processing...[/yellow]",
            AgentState.SPEAKING: "[blue]Speaking...[/blue]",
            AgentState.ERROR: "[red]Error[/red]",
        }
        console.print(f"Status: {status.get(state, str(state))}")

    def on_transcription(text: str):
        console.print(f"[cyan]You:[/cyan] {text}")

    def on_response(text: str):
        console.print(f"[green]Agent:[/green] {text}")

    agent.on_state_change(on_state_change)
    agent.on_transcription(on_transcription)
    agent.on_response(on_response)

    console.print("[dim]Voice agent ready. Speak to interact.[/dim]")

    # In a real implementation, this would connect to LiveKit
    # and process audio streams. For now, we demonstrate the API.
    console.print(
        "\n[yellow]Note:[/yellow] Full LiveKit integration requires a running "
        "LiveKit server. Use 'voice-agent demo' for a text-based demo."
    )


@app.command()
def demo() -> None:
    """Run a text-based demo of the voice agent."""
    if not check_environment():
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold blue]Voice Agent Demo Mode[/bold blue]\n\n"
            "Type messages to interact with the agent.\n"
            "Type 'quit' or 'exit' to stop.",
            border_style="blue",
        )
    )

    agent = create_agent()

    async def run_demo():
        # Greeting
        await agent.say("Hello! I'm your voice assistant. How can I help you today?")

        while True:
            try:
                user_input = console.input("[cyan]You:[/cyan] ")

                if user_input.lower() in ("quit", "exit"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                if not user_input.strip():
                    continue

                # Generate response
                response = await agent.llm.generate(user_input, agent.conversation)

                # Add to conversation history
                from voice_agent.models import ConversationTurn

                agent.conversation.append(ConversationTurn(role="user", content=user_input))
                agent.conversation.append(ConversationTurn(role="assistant", content=response.text))

                console.print(f"[green]Agent:[/green] {response.text}")

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Goodbye![/dim]")
                break

    asyncio.run(run_demo())


@app.command()
def status() -> None:
    """Check the status of the voice agent components."""
    console.print(
        Panel(
            "[bold]System Status Check[/bold]",
            border_style="blue",
        )
    )

    # Check Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        console.print("[green]Gemini API Key:[/green] Configured")
    else:
        console.print("[red]Gemini API Key:[/red] Not configured")

    # Check LiveKit
    livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    console.print(f"[blue]LiveKit URL:[/blue] {livekit_url}")

    # Check model
    model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    console.print(f"[blue]Model:[/blue] {model}")

    # Check TTS voice
    voice = os.getenv("TTS_VOICE", "en-US-AriaNeural")
    console.print(f"[blue]TTS Voice:[/blue] {voice}")


@app.command()
def voices(
    language: str = typer.Argument("en", help="Language code to filter voices"),
) -> None:
    """List available TTS voices."""
    from voice_agent.tts import TextToSpeech

    console.print(f"[bold]Available voices for '{language}':[/bold]\n")

    async def list_voices():
        voices = await TextToSpeech.list_voices(language)

        if not voices:
            console.print("[yellow]No voices found or Edge TTS not available.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Voice Name", style="cyan")
        table.add_column("Gender")
        table.add_column("Locale")

        for voice in voices[:20]:  # Limit to 20
            table.add_row(
                voice.get("ShortName", ""),
                voice.get("Gender", ""),
                voice.get("Locale", ""),
            )

        console.print(table)

    asyncio.run(list_voices())


@app.command()
def speak(
    text: str = typer.Argument(..., help="Text to speak"),
    voice: str = typer.Option(
        "en-US-AriaNeural",
        "--voice",
        "-v",
        help="TTS voice to use",
    ),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output audio file (MP3)"),
    ] = None,
) -> None:
    """Synthesize speech from text."""
    from voice_agent.models import TTSConfig
    from voice_agent.tts import create_tts

    console.print(f"[dim]Synthesizing with voice: {voice}[/dim]")

    tts = create_tts(TTSConfig(voice=voice))

    async def synthesize():
        audio = await tts.synthesize(text)

        if not audio:
            console.print("[red]Failed to synthesize audio.[/red]")
            return

        if output:
            output.write_bytes(audio)
            console.print(f"[green]Audio saved to:[/green] {output}")
        else:
            console.print(f"[green]Synthesized {len(audio)} bytes of audio.[/green]")
            console.print("[dim]Use --output to save to a file.[/dim]")

    asyncio.run(synthesize())


if __name__ == "__main__":
    app()
