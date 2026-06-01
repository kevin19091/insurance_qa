"""Evaluation utilities using RAGAS."""

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig


def load_eval_dataset(path: Path) -> pd.DataFrame:
    raise NotImplementedError


def run_evaluation(
    config: PipelineConfig, eval_df: pd.DataFrame, output_dir: Path
) -> dict[str, float]:
    raise NotImplementedError
