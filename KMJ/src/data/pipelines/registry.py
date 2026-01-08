from __future__ import annotations
from typing import Any, Dict, Type, Callable
from src.data.pipelines.base import DataPipelineBase

PIPELINE_REGISTRY: Dict[str, Type[DataPipelineBase]] = {}

def register_pipeline(name: str) -> Callable[[Type[DataPipelineBase]], Type[DataPipelineBase]]:
    def deco(cls: Type[DataPipelineBase]) -> Type[DataPipelineBase]:
        if name in PIPELINE_REGISTRY and PIPELINE_REGISTRY[name] is not cls:
            raise KeyError(f"Duplicate pipeline: {name}")
        PIPELINE_REGISTRY[name] = cls
        return cls
    return deco

def bootstrap_pipelines() -> list[str]:
    """
    Discover & import pipeline modules to trigger @register_pipeline decorators.
    Called by `src.bootstrap.bootstrap_registries()`.
    """
    from src.utils.registry_utils import autodiscover

    return autodiscover(
        "src.data.pipelines",
        exclude=("__init__", "base", "registry"),
        recursive=False,
    )
