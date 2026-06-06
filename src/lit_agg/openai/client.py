"""Thin wrapper around the OpenAI SDK for API key and base URL resolution."""

import os
import subprocess

from openai import OpenAI

from lit_agg.config import Config

API_BASE_URL_ENVS = ("LIT_AGG_OPENAI_BASE_URL", "OPENAI_BASE_URL")
OPENAI_KEY_ENVS = ("OPENAI_API_KEY",)
PROXY_KEY_ENVS = ("PI_PROXY_API_KEY", "SHOPIFY_AI_PROXY_TOKEN", "SHOPIFY_PROXY_KEY")


def _normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    return base_url.rstrip("/")


def _resolve_base_url(config: Config) -> str | None:
    if config.openai_base_url:
        return _normalize_base_url(config.openai_base_url)

    for env_name in API_BASE_URL_ENVS:
        value = os.environ.get(env_name)
        if value:
            return _normalize_base_url(value)

    return None


def _env_key(env_names: tuple[str, ...]) -> str | None:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _env_key_for_base_url(base_url: str | None) -> str | None:
    """Find an API key from environment variables.

    If no custom base URL is configured, only use OPENAI_API_KEY so the OpenAI
    SDK does not accidentally send an internal proxy token to the public OpenAI
    API. If a custom base URL is configured, also accept common internal proxy
    key env vars.
    """
    if base_url:
        return _env_key(OPENAI_KEY_ENVS + PROXY_KEY_ENVS)
    return _env_key(OPENAI_KEY_ENVS)


def get_client(config: Config, api_key: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client with key and base URL resolution.

    Base URL priority:
    1. openai_base_url in config
    2. LIT_AGG_OPENAI_BASE_URL env var
    3. OPENAI_BASE_URL env var
    4. OpenAI SDK default endpoint

    API key priority:
    1. Explicit api_key argument (from --api-key CLI flag)
    2. api_key_command from config (runs shell command to get a fresh key)
    3. Environment variable matching the configured endpoint
    """
    base_url = _resolve_base_url(config)

    resolved_key = api_key
    if not resolved_key and config.api_key_command:
        result = subprocess.run(
            config.api_key_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        resolved_key = result.stdout.strip()

    if not resolved_key:
        resolved_key = _env_key_for_base_url(base_url)

    if not resolved_key:
        if _env_key(PROXY_KEY_ENVS):
            raise ValueError(
                "Found an internal proxy API key, but no custom OpenAI-compatible "
                "base URL is configured. Set LIT_AGG_OPENAI_BASE_URL or "
                "openai_base_url in your local config."
            )
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY, pass --api-key, or configure "
            "api_key_command in your local config."
        )

    if base_url:
        return OpenAI(api_key=resolved_key, base_url=base_url)
    return OpenAI(api_key=resolved_key)
