"""Configuration loading with YAML override support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _default_profiles() -> dict[str, Any]:
    return {
        "physics": {
            "description": (
                "Theoretical physics papers relevant to holography and gauge/gravity "
                "duality, disorder and transport, charged systems, strange metals, "
                "quantum field theory, and nuclear theory or adjacent methods. Prefer "
                "papers with concrete physical models, analytic insight, or conceptual "
                "relevance. Deprioritize generic ML papers unless directly relevant to physics."
            ),
            "default_categories": ["hep-th", "cond-mat.str-el", "nucl-th"],
            "weekly_top_papers": 10,
            "weekly_max_candidates": 100,
        },
        "holography-transport": {
            "description": (
                "Focused high-energy and condensed matter theory papers on holography, "
                "gauge/gravity duality, AdS/CMT, momentum relaxation, disorder, charge "
                "and heat transport, strange metals, hydrodynamics, quantum chaos, and "
                "black brane models. Prefer explicit models or calculations of conductivity, "
                "diffusion, relaxation, or transport mechanisms. Deprioritize unrelated "
                "formal high-energy theory or generic condensed matter papers."
            ),
            "default_categories": ["hep-th", "cond-mat.str-el", "gr-qc", "cond-mat.stat-mech"],
            "weekly_top_papers": 10,
            "weekly_max_candidates": 150,
        },
        "nuclear-engineering": {
            "description": (
                "Papers useful for a theoretical physicist exploring nuclear engineering: "
                "reactor physics, radiation transport and detection, nuclear materials, "
                "fuel cycles, safety, thermal hydraulics, fusion/fission systems, nuclear "
                "data, and computational methods for applied nuclear systems. Include "
                "nuclear theory when it connects to applied nuclear technology. Deprioritize "
                "pure collider phenomenology or nuclear theory with no engineering link."
            ),
            "default_categories": [
                "nucl-th",
                "nucl-ex",
                "physics.ins-det",
                "physics.app-ph",
                "physics.comp-ph",
                "physics.plasm-ph",
            ],
            "weekly_top_papers": 10,
            "weekly_max_candidates": 150,
        },
        "ai-research-tools": {
            "description": (
                "AI and machine learning papers relevant to research tooling and scientific "
                "workflows: LLM agents, tool use, retrieval/RAG, literature discovery, code "
                "generation, evaluation methods, scientific reasoning, and AI-for-science "
                "systems. Prefer practical methods, careful evaluations, and tools that could "
                "improve research or software workflows. Deprioritize generic benchmark chasing."
            ),
            "default_categories": ["cs.AI", "cs.CL", "cs.LG", "cs.SE", "stat.ML"],
            "weekly_top_papers": 10,
            "weekly_max_candidates": 150,
        },
        "statistics": {
            "description": (
                "Statistical methodology and theory papers relevant to modern data science: "
                "Bayesian inference, uncertainty quantification, hierarchical models, "
                "nonparametrics, high-dimensional statistics, experimental design, missing "
                "data, measurement, robust inference, computation, and interpretable methods. "
                "Prefer papers with clear methodological contributions, practical estimators, "
                "or theory that changes how applied statistical work should be done."
            ),
            "default_categories": ["stat.ME", "stat.TH", "stat.CO", "stat.ML", "math.ST"],
            "weekly_top_papers": 10,
            "weekly_max_candidates": 150,
        },
        "causal-inference": {
            "description": (
                "Causal inference papers relevant to applied data science and econometrics: "
                "potential outcomes, DAGs and graphical models, treatment effects, experiments "
                "and A/B testing, instrumental variables, difference-in-differences, regression "
                "discontinuity, synthetic controls, causal discovery, causal ML, policy evaluation, "
                "sensitivity analysis, interference, and transportability. Prefer papers with "
                "clear identification assumptions, estimators, diagnostics, or applied guidance."
            ),
            "default_categories": ["stat.ME", "stat.ML", "stat.AP", "econ.EM", "cs.LG"],
            "weekly_top_papers": 10,
            "weekly_max_candidates": 150,
        },
    }


@dataclass
class Config:
    default_categories: list[str] = field(
        default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL"]
    )
    max_papers: int = 20
    batch_size: int = 10
    screening_batch_size: int = 20
    digest_max_candidates: int = 100
    digest_top_papers: int = 10
    summarize_model: str = "gpt-4o-mini"
    rank_model: str = "gpt-4o-mini"
    screen_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    api_key_command: str | None = None
    default_profile: str | None = None
    profiles: dict[str, Any] = field(default_factory=_default_profiles)


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
