"""
Entry point.

입력:
- --config: config YAML path
- --mode: pretrain|train_predict|predict
- --checkpoint: mode=predict일 때 필수

출력/부작용:
- run_dir 생성 + 로그 기록
- (train_predict) submission csv 생성
"""

import argparse
import ast
from omegaconf import OmegaConf

from src.utils import Setting, Logger
from src.bootstrap import bootstrap_registries
from src.factories.problem_factory import ProblemFactory
from src.factories.engine_factory import EngineFactory


# --------------------------------------------------
# config load (CLI > YAML)
# --------------------------------------------------
def load_and_merge_config():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # 최소 공통 옵션
    arg("--config", type=str, required=True)
    arg("--mode", type=str, default=None)  # pretrain|train_predict|predict
    arg("--checkpoint", type=str, default=None)
    arg("--seed", type=int, default=None)
    arg("--device", type=str, default=None)
    arg("--model", type=str, default=None)
    arg("--wandb", type=ast.literal_eval, default=None)
    arg("--run_name", type=str, default=None)

    cli_args = parser.parse_args()

    cfg = OmegaConf.load(cli_args.config)

    # CLI override (None 무시)
    cli_cfg = OmegaConf.create(vars(cli_args))
    for k, v in cli_cfg.items():
        if v is not None:
            cfg[k] = v

    return cfg


# --------------------------------------------------
# runtime infer (최소: pipeline만)
# engine.type은 config에 반드시 명시
# --------------------------------------------------
def infer_runtime_config(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # mode (SSoT)
    if "mode" not in cfg or not cfg.mode:
        cfg.mode = "train_predict"
    cfg.mode = str(cfg.mode).lower()
    if cfg.mode not in ("pretrain", "train_predict", "predict"):
        raise ValueError("cfg.mode must be one of: pretrain|train_predict|predict")

    # engine.type 필수
    if "engine" not in cfg or "type" not in cfg.engine:
        raise ValueError("engine.type must be specified in config")

    # model/model_args 정합성
    if "model" not in cfg or not cfg.model:
        raise ValueError("cfg.model is required")

    if "model_args" not in cfg or cfg.model not in cfg.model_args:
        raise ValueError("model_args must contain selected model config")

    # Problem은 name-only (너가 방금 바꾼 정책)
    if "problem" not in cfg or not cfg.problem.get("name"):
        raise ValueError("cfg.problem.name is required")

    # Pipeline은 data.pipeline SSoT
    if "data" not in cfg or not cfg.data.get("pipeline"):
        raise ValueError("cfg.data.pipeline is required")

    # Train section is mandatory (no legacy upgrade)
    if "train" not in cfg or cfg.train is None:
        raise ValueError("cfg.train is required (train.* is the single source of truth)")

    return cfg


# --------------------------------------------------
# normalize (실행용 cfg: 삭제만)
# - 값은 바꾸지 않는다
# --------------------------------------------------
def normalize_config(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    model = cfg.model
    engine_type = cfg.engine.type
    pipeline = cfg.data.pipeline

    # model_args: 선택 모델만 남김
    cfg.model_args = {model: cfg.model_args[model]}

    # data_args: 선택 pipeline만 남김 (존재하면)
    if "data_args" in cfg and pipeline in cfg.data_args:
        cfg.data_args = {pipeline: cfg.data_args[pipeline]}

    # mode에 따라 정규화(키 삭제는 최소화)
    if cfg.mode == "predict":
        # 학습 관련 섹션은 남겨도 무방하지만, 의미 없는 것들은 제거
        for k in ["optimizer", "lr_scheduler", "metrics", "loss"]:
            cfg.pop(k, None)

    # engine별 의미없는 섹션 제거
    if engine_type == "sklearn":
        for k in ["optimizer", "lr_scheduler", "metrics", "loss", "dataloader"]:
            cfg.pop(k, None)

    elif engine_type == "torch":
        cfg.pop("stratifiedkfold", None)

    elif engine_type == "recbole":
        # recbole이 자체적으로 학습/최적화 파라미터를 관리한다는 전제
        for k in ["optimizer", "lr_scheduler", "metrics", "loss", "dataloader"]:
            cfg.pop(k, None)

    return cfg


# --------------------------------------------------
# wandb (predict도 가능)
# --------------------------------------------------
def setup_wandb(cfg):
    if not cfg.get("wandb", False):
        for k in ["wandb_project", "run_name", "memo"]:
            cfg.pop(k, None)
        return None

    import wandb
    wandb.init(
        project=cfg.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.run_name if cfg.run_name else None,
        notes=cfg.memo if "memo" in cfg else None,
        tags=[cfg.model, cfg.engine.type, cfg.data.pipeline],
        resume="allow",
    )
    cfg.run_href = wandb.run.get_url()
    return wandb


# --------------------------------------------------
# main
# --------------------------------------------------
def main(cfg):
    bootstrap_registries()
    # 0) seed
    Setting.seed_everything(cfg.seed)

    # 1) run dir / logger는 train/predict 모두 생성
    setting = Setting()

    # run base dir: train.run_dir 우선, 없으면 saved/runs
    train_cfg = cfg.get("train", {})
    run_base_dir = train_cfg.get("run_dir", "saved/runs")

    run_dir = setting.get_run_dir(
        base_dir=run_base_dir,
        model=cfg.model,
        engine_type=cfg.engine.type,
        run_name=cfg.get("run_name"),
    )

    # runtime path resolution (cfg cannot reference Setting directly)
    try:
        if "recbole" in cfg:
            wd = cfg.recbole.get("work_dir", None)
            if not wd or (isinstance(wd, str) and "${setting.run_dir}" in wd):
                wd = (wd or "${setting.run_dir}/recbole").replace("${setting.run_dir}", run_dir)
                cfg.recbole["work_dir"] = wd
    except Exception:
        pass

    logger = Logger(cfg, run_dir)
    logger.save_args()

    # 2) Problem → DataBundle
    problem = ProblemFactory.build(cfg)
    data_bundle = problem.run()

    # 3) Engine
    engine = EngineFactory.build(cfg, logger, setting)

    # 4) mode routing
    if cfg.mode == "pretrain":
        tr = engine.fit(data_bundle)
        logger.log_predict_info({"mode": "pretrain", "checkpoint_path": tr.get("checkpoint_path") or ""})
        return

    if cfg.mode == "train_predict":
        tr = engine.fit(data_bundle)
        ckpt = tr.get("checkpoint_path")
        if not ckpt:
            raise ValueError("engine.fit() must return checkpoint_path for mode=train_predict")
        preds = engine.predict(data_bundle, checkpoint=ckpt)

    elif cfg.mode == "predict":
        if not cfg.checkpoint:
            raise ValueError("cfg.checkpoint is required for mode=predict")
        preds = engine.predict(data_bundle, checkpoint=cfg.checkpoint)

    # 5) optional evaluation (for wandb logging / sanity check)
    try:
        metrics = problem.evaluate_preds(preds, cfg, data_bundle)
        if metrics:
            logger.log_valid_metrics(metrics, step=0)
    except Exception as e:
        # evaluation shouldn't break submission run
        logger.log_predict_info({"stage": "eval_failed", "error": str(e)})

    # 6) save submission (Problem이 정책 결정, 경로 return)
    output_path = problem.save_submission(preds, cfg, setting, data_bundle)

    # 7) predict run logging
    logger.log_predict_info({
        "mode": cfg.mode,
        "checkpoint": cfg.checkpoint if cfg.mode == "predict" else "",
        "output_path": output_path or "",
        "engine": cfg.engine.type,
        "model": cfg.model,
        "pipeline": cfg.data.pipeline,
        "run_dir": run_dir,
    })

    # 8) artifact 기록(고도화는 나중에)
    if output_path:
        logger.log_artifact(output_path, name="submission")


if __name__ == "__main__":
    cfg = load_and_merge_config()
    cfg = infer_runtime_config(cfg)
    cfg = normalize_config(cfg)

    wandb_run = setup_wandb(cfg)

    print(OmegaConf.to_yaml(cfg))

    main(cfg)

    if wandb_run:
        wandb_run.finish()
