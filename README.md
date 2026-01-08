### RecSys RRS Baseline (MovieLens Seq Top-K)

이 레포는 **설정 파일(YAML) 기반으로** `Problem → DataPipeline → Engine`을 조립해 학습/예측/제출 파일 생성을 수행합니다.  
현재 기본 타겟은 **MovieLens 기반 Seq Top-K 제출** 플로우입니다.

추가 문서:
- `docs/ARCHITECTURE.md`: Problem→Pipeline→Engine 구조/Contract
- `docs/CONFIG_REFERENCE.md`: config 규칙/키 레퍼런스
- `docs/RUNBOOK.md`: 실행 런북(커맨드/산출물)

---

### 폴더 구조

- **`main.py`**: 실행 진입점(CLI). config 로드/정규화 후 실행
- **`config/`**: 실행 설정(YAML)
  - 예) `recbole_LGCN.yaml`, `recbole_RecVAE.yaml`
- **`data/`**: 데이터(예: `data/movielens/train/train_ratings.csv`)
- **`results/`**: 예시 결과(csv)
- **`src/`**: 핵심 코드
  - **`src/bootstrap.py`**: 레지스트리 부트스트랩(등록 트리거 단일 진입점)
  - **`src/problems/`**: 문제 정의(데이터 파이프라인 선택, 제출 정책)
  - **`src/data/`**
    - `pipelines/`: 데이터 로드/전처리/번들 생성
    - `loaders/`: raw 로더(디렉토리/파일 단위)
    - `transforms/`: (확장용) 변환 레지스트리
  - **`src/engines/`**: 학습/예측 엔진(torch/sklearn/recbole)
  - **`src/models/`**: 엔진별 recipe(모델 조립/overrides 생성 등)
  - **`src/utils/`**
    - `logger.py`: run 단위 로깅(파일 + wandb)
    - `cfg_utils.py`: cfg 타입 혼재(dict/DictConfig) 안전 접근 유틸

---

### 실행 방법(Quickstart)

#### 1) 설치

- **Windows 권장**

```bash
py -m pip install -r requirements.txt
```

- **Linux / macOS**

```bash
python -m pip install -r requirements.txt
```

- **RecBole 엔진을 쓰려면 추가 설치가 필요할 수 있습니다**

```bash
py -m pip install recbole
```

#### 2) MovieLens Seq Top-K (RecBole 예시)

```bash
py main.py --config config/recbole_LGCN.yaml
```

예측만 수행(체크포인트 지정):

```bash
py main.py --config config/recbole_LGCN.yaml --predict true --checkpoint "<ckpt_path>"
```

#### 3) MovieLens Seq Top-K (Torch S3Rec/SASRec 예시)

```bash
py main.py --config config/torch_S3Rec_seqtopn.yaml
```

#### 4) (선택) Torch S3Rec 프리트레인 → 파인튜닝 흐름

- 프리트레인(체크포인트만 생성, 제출 스킵):

```bash
py main.py --config config/torch_S3Rec_pretrain.yaml
```

- 파인튜닝에서 프리트레인 가중치 로드:
  - `config/torch_S3Rec_seqtopn.yaml`의 `train.pretrained_checkpoint`에 프리트레인 run의 `last.pt` 경로를 지정

---

### Config 규칙(중요)

#### 공통 키

- **`engine.type`**: 엔진 선택 키 (예: `recbole`, `torch`, `sklearn_topn`, `sklearn_regression`)
- **`problem.name`**: 문제 선택 키 (예: `movies_seq_topn`)
- **`data.pipeline`**: 파이프라인 선택 키 (예: `seq_topn_ml_v1`)
- **`model`**, **`model_args`**: 모델 이름과 하이퍼파라미터

#### Train 섹션(통일 스키마)

- **SSoT**: `train.*` (레거시 승격/호환 없음)
  - 레시피 전용 하이퍼는 **`recipe_args`** 로 분리해서 관리하는 걸 권장합니다.

예시(일부):

- `train.run_dir`: run 산출물 기본 디렉토리 (기본 `saved/runs`)
- `train.submit_dir`: 제출 파일 저장 디렉토리 (기본 `saved/submit`)
- `train.epochs`, `train.batch_size`, `train.lr`, `train.topk` 등

#### RecBole work_dir

- `recbole.work_dir`이 비어있거나 `${setting.run_dir}` 템플릿이면 런타임에 자동 해석됩니다.
  - 기본: `<run_dir>/recbole`

---

### 로깅/아웃풋(표준)

`train.run_dir` 하위에 run 디렉토리가 생성되고, 그 안에 로그가 쌓입니다.

- **`config.yaml`**, **`config_resolved.yaml`**
- **`events.jsonl`**: metrics/predict/artifact 이벤트 스트림(표준)
- **`predict_info.txt`**: 사람이 보기 쉬운 예측 로그
- **`artifacts.txt`**: artifact 인덱스
- torch 체크포인트 예시: `last.pt`

---

### 제출 포맷(MovieLens Seq Top-K)

- **샘플 제출 파일**: `data/movielens/eval/sample_submission.csv`
- **컬럼**: `user,item`
- **형태**:
  - 각 `user`마다 `K`개의 추천 `item`을 한 줄씩 기록합니다.
  - 즉, 총 행 수는 대략 `(#users * K)` 입니다.

이 레포의 `movies_seq_topn` 제출 저장 규칙:
- `Engine.predict()`는 **`preds: List[List[int]]`** 형태를 반환해야 합니다.
  - 바깥 list 길이 = 제출 대상 user 수
  - 안쪽 list 길이 = topK (예: 10)
- `Problem.save_submission()`이 `bundle.meta["submission"]["users"]` 순서에 맞춰 `(user, item)` row로 평탄화해서 저장합니다.
  - 구현: `src/problems/movies_seq_topn.py`
- **중요(대회 템플릿 방식)**:
  - 제출 대상 user/순서는 **`sample_submission.csv`를 SSoT로 사용**합니다(가능하면 이 파일을 그대로 채우는 방식).
  - 기본적으로 loader가 `<dataset.data_path>/../eval/sample_submission.csv`를 자동 탐색합니다.
  - 경로가 다르면 `dataset.sample_submission_path`로 직접 지정할 수 있습니다.
- 저장 위치:
  - 기본 `train.submit_dir` (없으면 `saved/submit`)
  - 파일명 기본값은 **`<model>.csv`** 입니다.

---

### 레지스트리/등록 정책

- **등록은 `src/bootstrap.py::bootstrap_registries()` 한 곳에서만 트리거**됩니다.
- `__init__.py` side-effect import로 등록하지 않습니다.

---

### 문제/파이프라인/엔진 개념(짧게)

- **Problem**
  - 어떤 `DataPipeline`을 사용할지 결정하고
  - `Engine.predict()` 결과를 **제출 포맷으로 저장**합니다.
- **DataPipeline**
  - raw 로드 → 정렬/전처리 → `DataBundle(train/valid/test/meta/schema)` 생성
- **Engine**
  - `fit(DataBundle)` / `predict(DataBundle)` 구현


