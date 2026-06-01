"""Test configuration and shared fixtures."""

import pytest

from src.config import PipelineConfig


@pytest.fixture
def default_config() -> PipelineConfig:
    return PipelineConfig()
