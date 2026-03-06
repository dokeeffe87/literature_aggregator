"""Thin wrapper around the Anthropic SDK for API key resolution."""

import subprocess

import anthropic

from lit_agg.config import Config


def get_client(config: Config, api_key: str | None = None) -> anthropic.Anthropic:
    """Create an Anthropic client with key resolution.

    Priority:
    1. Explicit api_key argument (from --api-key CLI flag)
    2. api_key_command from config (runs shell command to get fresh key)
    3. ANTHROPIC_API_KEY env var (SDK default)
    """
    if api_key:
        return anthropic.Anthropic(api_key=api_key)

    if config.api_key_command:
        result = subprocess.run(
            config.api_key_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        return anthropic.Anthropic(api_key=result.stdout.strip())

    # Fall back to SDK default (ANTHROPIC_API_KEY env var)
    return anthropic.Anthropic()
