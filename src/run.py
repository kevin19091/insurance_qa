"""CLI benchmark runner — ingest, evaluate, save artifacts for a milestone.

Usage:
    python -m src.run M0
    python -m src.run M0 --skip-eval
    python -m src.run M0 --output-dir benchmarks/M0
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config import PipelineConfig
from src.eval import run_eval
from src.pipeline.factory import build_generator, build_index, build_retriever

load_dotenv()

COST_PER_1K_TOKENS: dict[str, float] = {
    "gpt-4o-mini": 0.000150,
    "gpt-4o": 0.0025,
    "claude-3.5-sonnet": 0.003,
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    rate = COST_PER_1K_TOKENS.get(model, 0.000150)
    prompt_cost = (prompt_tokens / 1000) * rate
    completion_cost = (completion_tokens / 1000) * rate * 2
    return {
        "prompt_cost_usd": round(prompt_cost, 6),
        "completion_cost_usd": round(completion_cost, 6),
        "total_cost_usd": round(prompt_cost + completion_cost, 6),
    }


def _build_usage_log(generator: Any, model: str) -> dict[str, Any]:
    usage = generator.usage
    return {
        "model": model,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost": _estimate_cost(model, usage["prompt_tokens"], usage["completion_tokens"]),
    }


def run_benchmark(milestone: str, output_dir: Path, skip_eval: bool = False) -> dict[str, Any]:
    start_time = time.time()
    config_path = Path("benchmarks") / milestone / "config.yaml"

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig.from_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building index for {milestone}...")
    index = build_index(config)
    node_count = len(index.index_struct.nodes_dict) if hasattr(index, "index_struct") else 0
    print(f"Index built: {node_count} nodes")

    trace = {
        "milestone": milestone,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "config_path": str(config_path),
        "seed": config.seed,
        "prompt_version": config.prompt_version,
        "duration_seconds": 0.0,
        "index_node_count": node_count,
    }

    cost_log: dict[str, Any] = {
        "milestone": milestone,
        "model": config.llm.model,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": {"prompt_cost_usd": 0.0, "completion_cost_usd": 0.0, "total_cost_usd": 0.0},
    }

    eval_output = None

    if not skip_eval:
        retriever = build_retriever(index, config.retrieval.top_k)
        generator = build_generator(config)

        eval_output = run_eval(config_path, generator=generator, retriever=retriever)
        cost_log = _build_usage_log(generator, config.llm.model)

    trace["duration_seconds"] = round(time.time() - start_time, 2)

    with open(output_dir / "trace.json", "w") as f:
        json.dump(trace, f, indent=2, default=str)
    with open(output_dir / "cost_log.json", "w") as f:
        json.dump(cost_log, f, indent=2, default=str)
    if eval_output:
        eval_output["config_path"] = str(config_path)
        eval_output["config"] = config.model_dump()
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(eval_output, f, indent=2, default=str)

    return {
        "trace": trace,
        "config": config.model_dump(),
        "eval_results": eval_output,
        "cost_log": cost_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark milestone")
    parser.add_argument("milestone", help="Milestone name (e.g., M0)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: benchmarks/<milestone>)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval step (ingest only)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmarks") / args.milestone

    artifact = run_benchmark(args.milestone, output_dir, skip_eval=args.skip_eval)

    print("\n=== Benchmark Summary ===")
    print(f"  Milestone:   {args.milestone}")
    print(f"  Duration:    {artifact['trace']['duration_seconds']}s")
    print(f"  Model:       {artifact['config']['llm']['model']}")

    cost = artifact["cost_log"]["cost"]
    print(f"  Cost:        ${cost['total_cost_usd']:.6f}")

    if not args.skip_eval and artifact.get("eval_results"):
        scores = artifact["eval_results"]["scores"]
        print("\n  RAGAS Scores:")
        for metric, score in sorted(scores.items()):
            print(f"    {metric}: {score:.4f}")

    print(f"\n  Artifacts: {output_dir}/")


if __name__ == "__main__":
    main()
