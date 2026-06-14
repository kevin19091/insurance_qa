"""Tests for the Chroma inspect CLI."""
from pathlib import Path

import pytest


@pytest.mark.slow
class TestChromaInspect:
    def test_dumps_all_entries(self, tmp_path: Path) -> None:
        from src.config import PipelineConfig
        from src.pipeline.factory import build_index
        from src.chroma_inspect import dump_collection

        chroma_path = tmp_path / "chroma"
        config = PipelineConfig(storage={"chroma_path": str(chroma_path)})  # type: ignore[arg-type]
        build_index(config)

        entries = dump_collection(str(chroma_path))
        assert len(entries) > 0
        for entry in entries:
            assert "id" in entry
            assert "text" in entry
            assert "metadata" in entry
            assert len(entry["text"]) > 0

    def test_round_trip_through_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config import PipelineConfig
        from src.pipeline.factory import build_index

        chroma_path = tmp_path / "chroma_cli"
        config = PipelineConfig(storage={"chroma_path": str(chroma_path)})  # type: ignore[arg-type]
        build_index(config)

        monkeypatch.setattr("sys.argv", ["chroma_inspect", "--path", str(chroma_path)])
        from src.chroma_inspect import main
        main()
