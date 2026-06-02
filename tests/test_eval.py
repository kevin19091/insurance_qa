"""Tests for the eval harness."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

_OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))


class TestLoadQAPairs:
    def test_loads_ten_pairs(self) -> None:
        from src.eval import load_qa_pairs

        pairs = load_qa_pairs(Path("data/eval/qa.json"))
        assert len(pairs) == 10

    def test_each_pair_has_required_fields(self) -> None:
        from src.eval import load_qa_pairs

        pairs = load_qa_pairs(Path("data/eval/qa.json"))
        for pair in pairs:
            assert "question" in pair
            assert "reference_answer" in pair
            assert "source_pages" in pair
            assert len(pair["question"]) > 0
            assert len(pair["reference_answer"]) > 0


@pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
class TestRunEvalIntegration:
    def test_run_eval_returns_expected_shape(self) -> None:
        from src.eval import run_eval

        config_path = Path("benchmarks/M0/config.yaml")
        output = run_eval(config_path)

        assert "config_path" in output
        assert "config" in output
        assert "scores" in output
        assert "per_question" in output

        assert len(output["per_question"]) == 10
        for item in output["per_question"]:
            assert "question" in item
            assert "reference_answer" in item
            assert "generated_answer" in item
            assert "contexts" in item
            assert len(item["contexts"]) > 0

    def test_scores_are_float_between_zero_and_one(self) -> None:
        from src.eval import run_eval

        config_path = Path("benchmarks/M0/config.yaml")
        output = run_eval(config_path)

        for metric, score in output["scores"].items():
            assert 0.0 <= score <= 1.0, f"{metric} = {score} out of range"
