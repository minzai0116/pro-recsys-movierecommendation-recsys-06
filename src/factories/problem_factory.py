# src/factories/problem_factory.py
from __future__ import annotations

"""
ProblemFactory: cfg.problem.name → Problem 인스턴스 생성.

입력:
- cfg.problem.name: str (registry key)

출력:
- ProblemBase 구현체 인스턴스
"""

from typing import Any

from src.problems.base import ProblemBase
from src.problems.registry import PROBLEM_REGISTRY


class ProblemFactory:
    """
    v0 policy:
      - Problem is selected ONLY by cfg.problem.name
      - No pipeline->problem compatibility mapping
    """

    @classmethod
    def build(cls, cfg: Any) -> ProblemBase:
        if "problem" not in cfg:
            raise ValueError("cfg.problem is required (requires {name})")

        name = cfg.problem.get("name", None)
        if not name:
            raise ValueError("cfg.problem.name is required")

        if name not in PROBLEM_REGISTRY:
            raise ValueError(
                f"Unknown problem name={name}. Available={list(PROBLEM_REGISTRY.keys())}"
            )
        return PROBLEM_REGISTRY[name](cfg)
