"""Configuration loading with YAML override support."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    default_categories: list[str] = field(
        default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL"]
    )
    max_papers: int = 20
    batch_size: int = 10
    summarize_model: str = "claude-sonnet-4-6"
    rank_model: str = "claude-sonnet-4-6"
    api_key_command: str | None = None


def load_config(config_path: str | None = None) -> Config:
    """Load config from YAML file, falling back to defaults.

    Resolution order:
    1. Explicit path passed via --config
    2. ~/.config/lit-agg/config.yaml
    3. Shipped config.default.yaml
    4. Dataclass defaults
    """
    paths_to_try: list[Path] = []

    if config_path:
        paths_to_try.append(Path(config_path))
    paths_to_try.append(Path.home() / ".config" / "lit-agg" / "config.yaml")
    paths_to_try.append(Path(__file__).parent.parent.parent / "config.default.yaml")

    for path in paths_to_try:
        if path.is_file():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})

    return Config()
