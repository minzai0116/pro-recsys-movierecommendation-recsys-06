# src/engines/sklearn/registry.py
from src.engines.registry import register_engine

@register_engine("sklearn_regression")
def build_sklearn_regression_engine(cfg, logger, setting):
    from src.engines.sklearn.sklearn_regression_engine import SklearnRegressionEngine
    return SklearnRegressionEngine(cfg, logger, setting)

@register_engine("sklearn_topn")
def build_sklearn_topn_engine(cfg, logger, setting):
    from src.engines.sklearn.sklearn_topn_engine import SklearnTopNEngine
    return SklearnTopNEngine(cfg, logger, setting)
