"""Interest profile helpers for personalized literature digests."""

from dataclasses import dataclass, field
from typing import Any

from lit_agg.config import Config


@dataclass
class InterestProfile:
    """A named description of a user's research interests."""

    name: str
    description: str
    default_categories: list[str] = field(default_factory=list)
    weekly_top_papers: int | None = None
    weekly_max_candidates: int | None = None


class ProfileError(ValueError):
    """Raised when an interest profile cannot be resolved."""


def _optional_positive_int(name: str, field: str, value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ProfileError(f"Profile '{name}' {field} must be a positive integer.")
    return value


def _profile_from_mapping(name: str, data: Any) -> InterestProfile:
    if isinstance(data, str):
        return InterestProfile(name=name, description=data)

    if not isinstance(data, dict):
        raise ProfileError(f"Profile '{name}' must be a string or mapping.")

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ProfileError(f"Profile '{name}' must include a non-empty description.")

    categories = data.get("default_categories", [])
    if categories is None:
        categories = []
    if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
        raise ProfileError(f"Profile '{name}' default_categories must be a list of strings.")

    return InterestProfile(
        name=name,
        description=description.strip(),
        default_categories=[c.strip() for c in categories if c.strip()],
        weekly_top_papers=_optional_positive_int(
            name, "weekly_top_papers", data.get("weekly_top_papers")
        ),
        weekly_max_candidates=_optional_positive_int(
            name, "weekly_max_candidates", data.get("weekly_max_candidates")
        ),
    )


def resolve_profile(config: Config, name: str | None = None) -> InterestProfile:
    """Resolve an interest profile by name, config default, or single configured profile."""
    profiles = config.profiles or {}
    if not profiles:
        raise ProfileError(
            "No interest profiles configured. Add a profiles section to "
            "~/.config/lit-agg/config.yaml or pass --profile for an existing profile."
        )

    profile_name = name or config.default_profile
    if profile_name is None and len(profiles) == 1:
        profile_name = next(iter(profiles))
    if profile_name is None and "physics" in profiles:
        profile_name = "physics"

    if not profile_name:
        available = ", ".join(sorted(profiles))
        raise ProfileError(f"Choose an interest profile with --profile. Available: {available}")

    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ProfileError(f"Unknown profile '{profile_name}'. Available: {available}")

    return _profile_from_mapping(profile_name, profiles[profile_name])
