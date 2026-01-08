from __future__ import annotations
from typing import Callable, Dict, Protocol, Any

class EngineProtocol(Protocol):
    def fit(self) -> None: ...
    def predict(self) -> Any: ...

EngineBuilder = Callable[[Any, Any, Any], EngineProtocol]

ENGINE_REGISTRY: Dict[str, EngineBuilder] = {}

def register_engine(name: str):
    def deco(builder: EngineBuilder) -> EngineBuilder:
        if name in ENGINE_REGISTRY and ENGINE_REGISTRY[name] is not builder:
            raise KeyError(f"Duplicate engine: {name}")
        ENGINE_REGISTRY[name] = builder
        return builder
    return deco
