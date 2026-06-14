"""Tests for the CLI benchmark runner."""

from pathlib import Path

import pytest


class TestCostEstimate:
    def test_gpt4o_mini_cost(self) -> None:
        from src.run import _estimate_cost

        cost = _estimate_cost("gpt-4o-mini", 1000, 200)
        assert cost["prompt_cost_usd"] == pytest.approx(0.00015)
        assert cost["completion_cost_usd"] == pytest.approx(0.00006)
        assert cost["total_cost_usd"] == pytest.approx(0.00021)

    def test_unknown_model_falls_back(self) -> None:
        from src.run import _estimate_cost

        cost = _estimate_cost("unknown-model", 1000, 500)
        assert cost["total_cost_usd"] > 0


class TestUsageLog:
    def test_build_usage_log(self) -> None:
        from src.run import _build_usage_log

        class FakeGenerator:
            def __init__(self) -> None:
                self.usage = {
                    "prompt_tokens": 500,
                    "completion_tokens": 100,
                    "total_tokens": 600,
                }

        log = _build_usage_log(FakeGenerator(), "gpt-4o-mini")
        assert log["model"] == "gpt-4o-mini"
        assert log["prompt_tokens"] == 500
        assert log["completion_tokens"] == 100
        assert log["total_tokens"] == 600
        assert log["cost"]["total_cost_usd"] > 0


class TestOverrideParsing:
    def test_parse_single_override(self) -> None:
        from src.run import _parse_overrides

        parsed = _parse_overrides(["chunk.chunk_size=250"])
        assert parsed == {"chunk": {"chunk_size": 250}}

    def test_parse_multiple_overrides(self) -> None:
        from src.run import _parse_overrides

        parsed = _parse_overrides(
            ["chunk.strategy=sentence", "retrieval.top_k=10", "llm.model=gpt-4o"]
        )
        assert parsed == {
            "chunk": {"strategy": "sentence"},
            "retrieval": {"top_k": 10},
            "llm": {"model": "gpt-4o"},
        }

    def test_coerce_types(self) -> None:
        from src.run import _coerce

        assert _coerce("250") == 250
        assert _coerce("3.14") == 3.14
        assert _coerce("true") is True
        assert _coerce("false") is False
        assert _coerce("gpt-4o-mini") == "gpt-4o-mini"

    def test_apply_overrides_updates_config(self) -> None:
        from src.config import PipelineConfig
        from src.run import _apply_overrides

        config = PipelineConfig()
        _apply_overrides(config, {"chunk": {"chunk_size": 999}})
        assert config.chunk.chunk_size == 999

    def test_apply_overrides_unknown_section_exits(self) -> None:
        from src.config import PipelineConfig
        from src.run import _apply_overrides

        config = PipelineConfig()
        with pytest.raises(SystemExit):
            _apply_overrides(config, {"nonexistent": {"x": 1}})


@pytest.mark.slow
class TestRunBenchmark:
    def test_skip_eval_creates_artifacts(self, tmp_path: Path) -> None:
        from src.run import run_benchmark

        artifact = run_benchmark("M0", tmp_path, skip_eval=True)

        assert (tmp_path / "trace.json").exists()
        assert (tmp_path / "cost_log.json").exists()
        assert not (tmp_path / "eval_results.json").exists()

        assert artifact["trace"]["milestone"] == "M0"
        assert artifact["eval_results"] is None

    def test_run_benchmark_returns_expected_structure(self, tmp_path: Path) -> None:
        from src.run import run_benchmark

        artifact = run_benchmark("M0", tmp_path, skip_eval=False)

        assert "trace" in artifact
        assert "config" in artifact
        assert "eval_results" in artifact
        assert "cost_log" in artifact

        assert artifact["eval_results"] is not None
        assert "scores" in artifact["eval_results"]
        assert "per_question" in artifact["eval_results"]

        assert (tmp_path / "trace.json").exists()
        assert (tmp_path / "cost_log.json").exists()
        assert (tmp_path / "eval_results.json").exists()

    def test_trace_includes_timing(self, tmp_path: Path) -> None:
        from src.run import run_benchmark

        artifact = run_benchmark("M0", tmp_path, skip_eval=True)
        assert artifact["trace"]["duration_seconds"] > 0


class TestRebuildFlag:
    @pytest.mark.slow
    def test_rebuild_returns_same_node_count(self, tmp_path: Path) -> None:
        from src.run import run_benchmark

        first = run_benchmark("M0", tmp_path / "no_rebuild", skip_eval=True)
        second = run_benchmark("M0", tmp_path / "with_rebuild", skip_eval=True, rebuild=True)
        assert second["trace"]["index_node_count"] == first["trace"]["index_node_count"]
