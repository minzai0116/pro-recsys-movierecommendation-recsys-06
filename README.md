# Movie Recommendation

## 1. Introduction

### 1.1. 프로젝트 소개
MovieLens 데이터를 활용해 사용자의 시청 이력과 타임스탬프를 기반으로 다음에 볼 영화와 선호 영화 후보를 예측하는 추천 시스템 프로젝트입니다.

이번 프로젝트는 단순한 정적 선호 예측을 넘어, 시청 순서와 중간 누락 아이템까지 함께 고려해야 하는 실제 서비스에 가까운 문제 설정을 다뤘습니다. 그래서 `정적 취향`과 `시퀀셜 패턴`을 동시에 반영할 수 있도록 후보군 생성과 리랭킹을 분리한 2-Stage 추천 구조를 중심으로 실험했습니다.

### 1.2. Project Objective
- MovieLens 기반 Top-K 추천 성능 향상
- 정적 선호와 순차적 패턴을 함께 반영할 수 있는 구조 설계
- 메타데이터 정제와 Feature Engineering을 통한 추천 품질 개선
- 단일 모델 비교를 넘어, 후보군 생성과 리랭킹이 분리된 실전형 파이프라인 구축

### 1.3. 내가 담당한 역할
- 메타데이터 EDA
- 2-Stage 리랭킹 시스템 설계
- 추천 품질 향상을 위한 Feature Engineering 방향 설계

## 2. Key Ideas

### 2.1. 문제 특성
- 암시적 피드백 기반 추천 문제
- 마지막 아이템 예측뿐 아니라 중간 누락 아이템 복원까지 고려
- 정적 선호와 동적 시청 흐름을 함께 다뤄야 하는 난도 높은 환경

### 2.2. EDA 및 Feature Engineering
- 계절별 시청 패턴과 시간대 선호를 분석해 `season`, `dayparting` 피처를 설계
- 유저 과거 시청 장르 비율 피처 생성
- 영화 인기도 및 랭킹 피처 생성
- 감독/작가의 1:N 관계를 정제해 대표 인물 정보로 차원 축소
- 장르 TF-IDF를 적용해 흔한 장르 영향력을 낮추고 희귀 취향 신호를 강화
- 제목 패턴 분석과 정규표현식으로 연도 결측치 100% 복원

### 2.3. 핵심 인사이트
- 상위 1% 아이템이 전체 시청의 16.4%를 차지하는 강한 인기도 편향이 존재했습니다.
- 단순 시퀀셜 모델은 데이터 특성과 맞지 않았고, 전역 공출현을 잘 잡는 정적 모델이 더 강했습니다.
- 좋은 추천은 단일 모델 성능보다, 어떤 후보군을 만들고 어떤 맥락 피처로 재정렬하느냐에 더 크게 좌우됐습니다.

## 3. Modeling

### 3.1. 실험한 주요 모델
- `S3Rec`
- `EASE`
- `Multi-VAE`
- `CatBoost Ranker`
- Transformer Encoder 기반 실험 모델
- 앙상블 실험

### 3.2. 최종 전략
- `Stage 1`: `EASE + Multi-VAE`로 후보군 생성
- `Stage 2`: 메타데이터와 맥락 피처를 활용한 `CatBoost Ranker`로 재정렬

이 구조를 선택한 이유는 다음과 같습니다.
- EASE는 dense한 상호작용 환경에서 강한 전역 상관관계를 잘 포착했습니다.
- Multi-VAE는 비선형 취향 패턴을 보완해 후보 다양성을 늘렸습니다.
- CatBoost Ranker는 정제된 메타데이터와 컨텍스트 피처를 반영해 최종 순위를 안정화했습니다.

## 4. Results

### 4.1. 성능 요약

| 모델 | Public LB |
|------|-----------|
| S3Rec | 0.0559 |
| EASE (Base) | 0.1572 |
| EASE (Optuna Tuned) | 0.1599 |
| EASE + Multi-VAE | 0.1620 |
| 2-Stage (CatBoost Final) | 0.1670 |

### 4.2. 최종 성과
- 최종 아키텍처는 `2-Stage Re-ranking System`입니다.
- Stage 1에서 유저당 200개의 후보군을 생성하고, Stage 2에서 약 40개의 피처를 활용해 최종 Top-10을 선정했습니다.
- `Valid Recall@10 0.2012`, `Public LB 0.1670`으로 프로젝트 최고 성능을 달성했습니다.

## 5. Repository Guide

```text
pro-recsys-movierecommendation-recsys-06
├─ config/              # 모델별 실행 설정
├─ docs/                # 아키텍처, 설정, 실행 가이드
├─ src/                 # 공통 추천 프레임워크
├─ KMJ/                 # 개인 실험/확장 코드
├─ main.py              # 실행 진입점
└─ README.md
```

### 5.1. 내부 구조
- `src/problems/`: 문제 정의 및 제출 규칙
- `src/data/`: loader, pipeline, transform
- `src/engines/`: torch, sklearn, recbole 엔진
- `src/models/`: 모델별 recipe 및 구현
- `src/bootstrap.py`: 레지스트리 등록 진입점

추가 문서:
- `docs/ARCHITECTURE.md`
- `docs/CONFIG_REFERENCE.md`
- `docs/RUNBOOK.md`

## 6. How To Run

### 6.1. Install

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

### 6.4. Predict

```bash
python3 main.py --config config/recbole_LGCN.yaml --mode predict --checkpoint <ckpt_path>
```

## 7. Team

| 이름 | 역할 |
|------|------|
| 김태형 | 프레임워크 개발 및 유지 보수 |
| 김민재 | 메타데이터 EDA, 2-Stage 리랭킹 시스템 설계 |
| 석찬휘 | DL 모델 설계 및 실험 |
| 조형동 | DL 모델 설계 및 실험, EASE+BERT 앙상블 구현 |
| 최영진 | EDA, Feature Engineering, 모델 구현 및 테스트 |
