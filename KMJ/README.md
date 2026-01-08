# NewFrame: EASE 추천 시스템

## 개요

**EASE (Embarrassingly Shallow Autoencoders)** 모델을 기반으로 한 영화 추천 시스템입니다.

**현재 성능:**
- **검증 Recall@10**: 0.1895
- **실제 대회 점수**: 0.1572
- **목표**: 0.18+

## 특징

- ✅ **Closed-form Solution**: 수학적 역행렬 계산으로 매우 빠른 학습 (수 초 내 완료)
- ✅ **시간복잡도 최적화**: 희소 행렬(`scipy.sparse`) 활용, 벡터화된 배치 예측
- ✅ **데이터 누수 방지**: User-sequence split (모든 사용자 포함, 아이템만 마스킹)
- ✅ **대회 규칙 준수**: Normalized Recall@10 (분모: `min(10, |Ground Truth|)`)
- ✅ **중간 Recall 계산**: 예측 중간에 진행 상황 확인 및 로깅
- ✅ **WandB 통합**: 실험 추적 및 메트릭 시각화
- ✅ **하이퍼파라미터 최적화**: Optuna를 통한 Lambda 자동 탐색

## 설치

```bash
pip install -r requirements.txt
```

**주요 의존성:**
- `numpy`, `scipy` (희소 행렬 연산)
- `pandas` (데이터 처리)
- `tqdm` (진행바)
- `wandb` (실험 추적)
- `optuna` (하이퍼파라미터 최적화)
- `pyyaml` (설정 파일)

## 실행

### 1. 기본 학습 및 Validation

```bash
python main.py --config config.yaml
```

**실행 과정:**
1. 데이터 로드 및 User-Item 행렬 생성
2. User-sequence split (마지막 1개 + 중간 dropout 9개)
3. EASE 모델 학습 (Closed-form solution)
4. Validation Recall@10 계산
5. WandB에 메트릭 로깅

### 2. Test 예측 및 제출 파일 생성

`config.yaml`에서 `predict_test: true`, `retrain_on_full: true`로 설정 후 실행:

```bash
python main.py --config config.yaml
```

**출력:**
- `submission.csv`: 제출 형식 파일 (31,360명 × 10개 아이템)

### 3. Lambda 하이퍼파라미터 최적화

`config.yaml`에서 `optimize_lambda: true`로 설정:

```bash
python main.py --config config.yaml
```

Optuna가 베이지안 최적화로 최적 Lambda 값을 탐색합니다.

## 설정 파일 (config.yaml)

```yaml
# 시드
seed: 42

# 데이터 경로
data:
  train_path: "../RecSys_RRS_Framework/data/train/train_ratings.csv"

# Validation 설정
val_items_per_user: 10  # 사용자당 Validation 아이템 수
# - 마지막 1개는 반드시 포함
# - 나머지 9개는 중간에서 랜덤 선택 (실제 대회와 유사)

# 모델 하이퍼파라미터
model:
  lambda_reg: 100.0  # 정규화 파라미터 (50-200 권장)

# 예측 설정
prediction:
  batch_size: 1000  # 배치 크기
  check_interval: 100  # 중간 Recall 계산 주기 (명)

# 출력
output:
  submission_path: "submission.csv"

# 옵션
predict_test: true  # Test 예측 여부
retrain_on_full: true  # 전체 데이터로 재학습 여부
optimize_lambda: false  # Lambda 자동 최적화 여부

# WandB 설정
wandb:
  enabled: true
  project: "KMJ-movie-rec"
  entity: "timesmoker-ronaldo-s-iron-discipline"
  run_name: "ease_baseline"
```

## EASE 모델 설명

### 수학적 배경

**EASE (Embarrassingly Shallow Autoencoders)**는:
- **Closed-form solution**: 반복적 최적화 없이 수학적 역행렬 계산으로 학습
- **전역적 상관관계**: 아이템 간 복잡한 관계를 한 번에 학습
- **메모리 효율성**: 희소 행렬로 대규모 데이터 처리 가능

**학습 수식:**
```
G = X^T X  (아이템-아이템 공분산 행렬)
P = (G + λI)^(-1)  (정규화된 역행렬)
B = -P / diag(P)  (정규화)
B[i, i] = 0  (self-connection 제거)
```

**예측:**
```
scores = user_vector @ B  (벡터화된 행렬 곱셈)
```

### 구현 최적화

1. **희소 행렬 활용**: `scipy.sparse.csr_matrix`로 메모리 사용량 대폭 감소
2. **벡터화된 배치 예측**: 모든 사용자에 대해 한 번에 예측 (`predict_batch_vectorized`)
3. **효율적 Top-K 선택**: `np.argpartition`으로 빠른 상위 아이템 추출
4. **O(N^2) 병목 해결**: Set 자료구조로 사용자 검색 최적화

## 성능 분석

### 현재 성능
- **검증 Recall@10**: 0.1895
- **실제 대회 점수**: 0.1572
- **격차**: 약 0.03 (검증 방식 개선 필요)

### 검증 전략
- **User-sequence split**: 모든 사용자를 학습에 포함
- **마스킹 전략**: 마지막 1개 + 중간 dropout 9개 (실제 대회와 유사)
- **Normalized Recall@10**: `len(intersection) / min(10, |Ground Truth|)`

## 성능 향상 방안

### 1. 하이퍼파라미터 최적화
- Lambda 값 조정 (50-200 범위 탐색)
- Optuna를 통한 자동 최적화

### 2. 앙상블
- 여러 Lambda 값으로 학습 후 가중 평균
- 예상 효과: +0.01~0.02

### 3. 2-Stage Re-ranking
- EASE로 Top-100 후보 생성
- CatBoost/XGBoost Ranker로 Top-10 선택
- 메타데이터 피처 활용
- 예상 효과: +0.02~0.03

### 4. Co-visitation Matrix
- 아이템 간 동시 출현 패턴 활용
- 후보 생성 보강
- 예상 효과: +0.01~0.015

## 파일 구조

```
NewFrame/
├── src/
│   ├── __init__.py
│   ├── utils.py              # 시드 고정 등 유틸리티
│   ├── data_utils.py         # 데이터 로드, 행렬 생성, split
│   ├── ease.py               # EASE 모델 구현
│   ├── metrics.py            # Recall@K 계산
│   └── hyperparameter_tuning.py  # Optuna 최적화
├── main.py                   # 메인 실행 스크립트
├── config.yaml               # 설정 파일
├── requirements.txt          # 의존성
├── README.md                 # 이 파일
└── submission.csv            # 제출 파일 (생성됨)
```

## 참고 자료

- **논문**: [Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375)
- **대회 평가 지표**: Normalized Recall@10
  - 분자: 예측 Top-K와 Ground Truth의 교집합 크기
  - 분모: `min(K, |Ground Truth|)`
  - 참고: https://arxiv.org/pdf/1802.05814.pdf

## 로그 예시

```
✅ EASE 모델 학습 완료 (16.2초)
✅ Validation 예측 완료
✅ 최종 Validation Recall@10: 0.1895
  → 대회 최고 점수 0.17~0.18에 근접한 성능입니다! 🎉

✅ 전체 데이터로 재학습 완료
✅ Test 예측 완료
✅ 제출 파일 생성 완료: submission.csv
```

## 다음 단계

1. **Lambda 최적화**: Optuna로 최적 값 탐색
2. **앙상블**: 여러 Lambda 값 조합
3. **Re-ranking**: CatBoost Ranker 추가
4. **Co-visitation**: 아이템 관계 활용
