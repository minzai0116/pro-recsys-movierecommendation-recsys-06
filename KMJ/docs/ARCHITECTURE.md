### Architecture (Problem → DataPipeline → Engine)

이 레포는 **config(YAML)** 로 “문제 정의/데이터 파이프라인/학습 엔진”을 조립합니다.
핵심은 모든 구성요소가 **입·출력 Contract** 로 연결된다는 점입니다.

---

### 1) High-level flow

- **`main.py`**
  - config 로드/검증(`infer_runtime_config`)
  - run_dir 생성 + Logger 초기화
  - `Problem.run()`으로 `DataBundle` 생성
  - `Engine.fit()/predict()` 실행
  - `Problem.evaluate_preds()`(선택) → metrics logging
  - `Problem.save_submission()` → 제출 파일 저장

---

### 2) Contracts (핵심 규약)

#### DataBundle (`src/data/data_bundle.py`)

- **입력**: `train/valid/test` DataFrame + `schema` + `meta`
- **출력**: Engine이 consume 하는 표준 컨테이너

특히 `movies_seq_topn`에서는 meta가 중요합니다:
- `meta["submission"]["users"]`: 제출 user 순서 (SSoT: sample_submission 기반)
- `meta["user_seq"]`: user → item sequence (time 정렬)

#### Problem (`src/problems/base.py`)

- **입력**: cfg
- **출력**:
  - `run() -> DataBundle`
  - `save_submission(preds, ...) -> Optional[str]`
  - `evaluate_preds(preds, ...) -> dict|None` (선택)

#### DataPipeline (`src/data/pipelines/base.py`)

- **입력**: cfg + raw dict
- **출력**: `DataBundle`

#### Engine (`src/engines/core/engine_base.py`)

- **입력**: cfg + DataBundle
- **출력(Contract)**:
  - `fit(bundle) -> {"checkpoint_path": Optional[str], ...}`
  - `predict(bundle, checkpoint=...) -> preds`

---

### 3) Registries / bootstrap

등록(Decorator)은 import 시점에 발생하므로, **등록 트리거를 단일화**해야 합니다.

- **등록 트리거**: `src/bootstrap.py::bootstrap_registries()`
- 원칙: `__init__.py` side-effect import 금지
- 구현: `src/utils/registry_utils.py::autodiscover()`


