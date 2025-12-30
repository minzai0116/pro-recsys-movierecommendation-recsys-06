"""
`src.problems` package.

중요:
- 여기서 하위 모듈을 자동 import 하지 않습니다.
  (삭제된/실험용 모듈 때문에 엔트리포인트가 깨지는 것을 방지)
- Problem 등록은 `src.bootstrap.bootstrap_registries()` →
  `src.problems.registry.bootstrap_problems()`(autodiscover)에서 수행합니다.
"""

__all__: list[str] = []