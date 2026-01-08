# src/problems/registry.py
from __future__ import annotations

from typing import Callable, Dict, Type, Any

from src.problems.base import ProblemBase

PROBLEM_REGISTRY: Dict[str, Type[ProblemBase]] = {}
PIPELINE_TO_PROBLEM: Dict[str, str] = {}


def register_problem(
    name: str, *, pipelines: list[str] | None = None
) -> Callable[[Type[ProblemBase]], Type[ProblemBase]]:
    """
    Register a Problem class.

    - name: cfg.problem.name key
    - pipelines: optional legacy cfg.problem.pipeline keys mapping to this problem
    """
    def deco(cls: Type[ProblemBase]) -> Type[ProblemBase]:
        exist = PROBLEM_REGISTRY.get(name)
        if exist is not None and exist is not cls:
            raise KeyError(f"Duplicate problem registration: {name} -> {exist} vs {cls}")
        PROBLEM_REGISTRY[name] = cls

        if pipelines:
            for p in pipelines:
                mapped = PIPELINE_TO_PROBLEM.get(p)
                if mapped is not None and mapped != name:
                    raise KeyError(f"Duplicate pipeline fallback: {p} -> {mapped} vs {name}")
                PIPELINE_TO_PROBLEM[p] = name

        return cls
    return deco


def bootstrap_problems() -> list[str]:
    """
    Discover & import problem modules to trigger @register_problem decorators.
    Called by `src.bootstrap.bootstrap_registries()`.
    """
    from src.utils.registry_utils import autodiscover

    return autodiscover(
        "src.problems",
        exclude=("__init__", "base", "registry"),
        recursive=False,
    )