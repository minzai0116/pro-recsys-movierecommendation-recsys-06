### Config Reference (SSoT rules)

이 레포는 **레거시 승격/호환 없이** 아래 규칙을 강제합니다.

---

### 1) Top-level

- **`mode`**: `pretrain | train_predict | predict`
  - `pretrain`: 학습만 수행 (제출/예측 없음)
  - `train_predict`: 학습 → 예측 → 제출 생성
  - `predict`: 체크포인트 로드 → 예측 → 제출 생성
- **`checkpoint`**: `mode=predict`일 때 필수
- **`seed`**, **`device`**, **`run_name`**
- **`wandb`**, **`wandb_project`**, **`memo`**

---

### 2) Routing keys

- **`engine.type`**: `torch | recbole | sklearn_*`
- **`problem.name`**: 예) `movies_seq_topn`
- **`data.pipeline`**: 예) `seq_topn_ml_v1`
- **`recipe`**: (torch에서 주로 사용) 예) `s3rec_finetune`, `s3rec_pretrain`
- **`model`**, **`model_args`**: 모델 이름/구조 하이퍼

---

### 3) `train.*` (공통 SSoT)

`train.*`에는 **학습 루프/옵티마이저/런 경로**처럼 “공통”만 둡니다.

예:
- `train.run_dir`, `train.submit_dir`
- `train.epochs`, `train.batch_size`, `train.lr`, `train.topk`
- `train.weight_decay`, `train.adam_beta1`, `train.adam_beta2`
- `train.num_workers`

---

### 4) `recipe_args` (레시피 전용)

특정 레시피 전용 하이퍼는 `recipe_args`로 분리합니다.

예) `recipe: s3rec_pretrain`
- `recipe_args.mask_p`
- `recipe_args.aap_weight`, `mip_weight`, `map_weight`, `sp_weight`


