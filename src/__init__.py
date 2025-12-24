"""
`src` package.

중요:
- 패키지 import 자체가 부작용(side-effect)으로 실패하면 전체 실행이 깨집니다.
- 레지스트리 등록은 `src.bootstrap.bootstrap_registries()`에서 수행합니다.
"""

__all__: list[str] = []