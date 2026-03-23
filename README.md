# Movie Recommendation

## 1. 프로젝트 소개

MovieLens 데이터를 활용해 사용자의 시청 이력과 타임스탬프를 기반으로 다음에 볼 영화와 선호 영화 후보를 예측하는 추천 시스템 프로젝트입니다.

이 프로젝트는 단순한 정적 선호 예측을 넘어, 시청 순서와 중간 누락 아이템까지 함께 고려해야 하는 실제 서비스에 가까운 문제를 다룹니다. 그래서 정적 취향과 시퀀셜 패턴을 동시에 반영할 수 있도록 후보군 생성과 리랭킹을 분리한 2단계 추천 구조를 중심으로 실험을 진행했습니다.

## 2. 프로젝트 목표

- MovieLens 기반 Top-K 추천 성능 향상
- 정적 선호와 순차적 패턴을 함께 반영할 수 있는 구조 설계
- 메타데이터 정제와 Feature Engineering을 통한 추천 품질 개선
- 단일 모델 비교를 넘어, 후보군 생성과 리랭킹이 분리된 추천 파이프라인 구축

## 3. 문제 정의와 접근

### 3.1. 데이터 특성
- 암시적 피드백 기반 추천 문제입니다.
- 마지막 아이템 예측뿐 아니라, 중간에 누락된 아이템 복원까지 고려해야 합니다.
- 정적 선호와 동적 시청 흐름을 함께 다뤄야 하는 난도 높은 환경입니다.

### 3.2. EDA 및 Feature Engineering
- 계절별 시청 패턴과 시간대 선호를 분석해 `season`, `dayparting` 피처를 설계했습니다.
- 유저 과거 시청 장르 비율 피처를 생성했습니다.
- 영화 인기도 및 랭킹 피처를 생성했습니다.
- 감독과 작가의 1:N 관계를 정제해 대표 인물 정보로 차원을 축소했습니다.
- 장르 TF-IDF를 적용해 흔한 장르의 영향력을 낮추고 희귀 취향 신호를 강화했습니다.
- 제목 패턴 분석과 정규표현식으로 연도 결측치를 복원했습니다.

### 3.3. 모델링 전략
- `S3Rec`
- `EASE`
- `Multi-VAE`
- `CatBoost Ranker`
- Transformer Encoder 기반 실험 모델
- 앙상블 실험

최종적으로는 다음과 같은 2단계 구조를 선택했습니다.
- `Stage 1`: `EASE + Multi-VAE`로 후보군 생성
- `Stage 2`: 메타데이터와 맥락 피처를 활용한 `CatBoost Ranker`로 재정렬

이 구조는 전역 공출현 패턴과 비선형 취향 패턴을 함께 반영하고, 정제된 메타데이터를 통해 최종 순위를 안정적으로 조정하기 위한 선택이었습니다.

## 4. 실험 결과

### 4.1. 성능 요약

| 모델 | Public LB |
|------|-----------|
| S3Rec | 0.0559 |
| EASE (Base) | 0.1572 |
| EASE (Optuna Tuned) | 0.1599 |
| EASE + Multi-VAE | 0.1620 |
| 2-Stage (CatBoost Final) | 0.1670 |

### 4.2. 결과 요약
- 최종 아키텍처는 `2-Stage Re-ranking System`입니다.
- Stage 1에서 유저당 200개의 후보군을 생성하고, Stage 2에서 약 40개의 피처를 활용해 최종 Top-10을 선정했습니다.
- `Valid Recall@10 0.2012`, `Public LB 0.1670`으로 프로젝트 최고 성능을 달성했습니다.

## 5. 저장소 구조

```text
pro-recsys-movierecommendation-recsys-06
├─ config/              # 모델별 실행 설정
├─ docs/                # 아키텍처, 설정, 실행 가이드
├─ src/                 # 공통 추천 프레임워크
├─ KMJ/                 # 개인 실험/확장 코드
├─ main.py              # 실행 진입점
└─ README.md
```

### 5.1. 내부 구성
- `src/problems/`: 문제 정의 및 제출 규칙
- `src/data/`: loader, pipeline, transform
- `src/engines/`: torch, sklearn, recbole 엔진
- `src/models/`: 모델별 recipe 및 구현
- `src/bootstrap.py`: 레지스트리 등록 진입점

참고 문서:
- `docs/ARCHITECTURE.md`
- `docs/CONFIG_REFERENCE.md`
- `docs/RUNBOOK.md`

## 6. 실행 방법

### 6.1. 설치

```bash
python3 -m pip install -r requirements.txt
```

### 6.2. RecBole 기반 실행 예시

```bash
python3 main.py --config config/recbole_LGCN.yaml
```

### 6.3. Torch 기반 실행 예시

```bash
python3 main.py --config config/torch_S3Rec_seqtopn.yaml
```

### 6.4. 예측

```bash
python3 main.py --config config/recbole_LGCN.yaml --mode predict --checkpoint <ckpt_path>
```

## 7. 팀

| 이름 | 역할 |
|------|------|
| 김태형 | 프레임워크 개발 및 유지 보수 |
| 김민재 | 메타데이터 EDA, 2-Stage 리랭킹 시스템 설계 |
| 석찬휘 | DL 모델 설계 및 실험 |
| 조형동 | DL 모델 설계 및 실험, EASE+BERT 앙상블 구현 |
| 최영진 | EDA, Feature Engineering, 모델 구현 및 테스트 |
