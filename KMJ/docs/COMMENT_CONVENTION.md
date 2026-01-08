### 주석/Docstring 규약 (입·출력 중심)

이 레포의 주석은 “왜 이렇게 구현했는지”보다 **무엇을 입력으로 받고 무엇을 출력으로 내는지(Contract)** 를 우선 설명합니다.

---

### 1) 모듈 docstring (파일 최상단)

각 `*.py` 파일 맨 위에 다음을 권장합니다.

- **역할**: 이 모듈이 담당하는 책임
- **입력**: 어떤 데이터/cfg/meta를 받는지
- **출력**: 어떤 타입/형식으로 반환하는지
- **관련 설정 키**: 자주 쓰는 `cfg.*` 키

예시:

```python
"""
MovieLens sequential loader.

입력:
- cfg.dataset.data_path: str

출력:
- dict[str, Any] (keys: ratings, sample_submission, item2attributes, ...)
"""
```

---

### 2) 클래스 docstring

클래스는 다음을 명시합니다.

- **책임(Responsibility)**
- **입력/출력 계약**: public method의 입력/출력
- **상태(state)**: 내부에 저장하는 주요 필드

---

### 3) 함수/메서드 docstring

함수는 “입력/출력”을 우선으로, 필요 시 예외/부작용을 덧붙입니다.

권장 구성:
- **한 줄 요약**
- **입력**: 타입/형식/필수 키
- **출력**: 타입/형식/shape/정렬 규칙
- **예외**: 어떤 조건에서 raise 하는지
- **부작용**: 파일 생성/디렉터리 생성/로그 기록 등

---

### 4) 라인 주석 사용 기준

라인 주석은 아래 케이스에만 사용합니다.

- **실수하기 쉬운 규칙**(예: “sample_submission이 SSoT”, “seen item masking”)
- **경계 조건**(예: empty/None 처리)
- **대회 제출 포맷**처럼 외부 규약

그 외 설명은 docstring으로 옮깁니다.


