### Runbook (MovieLens Seq Top-K)

이 문서는 “바로 실행” 관점의 런북입니다.

---

### 1) Train → Predict → Submit (기본)

```bash
py main.py --config config/torch_S3Rec_seqtopn.yaml
```

---

### 2) Pretrain only (checkpoint 생성)

```bash
py main.py --config config/torch_S3Rec_pretrain.yaml
```

---

### 3) Predict only (checkpoint 지정)

`mode: predict` + `checkpoint`가 필요합니다.

```bash
py main.py --config config/torch_S3Rec_seqtopn.yaml --mode predict --checkpoint "saved/runs/<run>/last.pt"
```

---

### 4) 산출물 위치

각 실행(run)은 `train.run_dir` 하위에 디렉토리를 만들고 파일을 기록합니다:
- `config.yaml`, `config_resolved.yaml`
- `events.jsonl` (metrics/predict/artifact)
- torch: `last.pt`

제출 파일은 기본적으로 `train.submit_dir` 하위에 저장됩니다.


