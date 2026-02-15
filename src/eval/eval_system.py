from __future__ import annotations

import argparse
import json
import time

import torch

from src.config import ProjectConfig, ensure_project_dirs
from src.inference.generate import run_inference


def run_system_eval(config: ProjectConfig, query: str) -> dict:
    ensure_project_dirs(config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _ = run_inference(query, config=config, max_new_tokens=120)
    elapsed = time.perf_counter() - started

    gpu_peak_mb = 0.0
    if torch.cuda.is_available():
        gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    result = {"latency_seconds": elapsed, "gpu_peak_memory_mb": gpu_peak_mb}
    (config.reports_dir / "system_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run system evaluation.")
    parser.add_argument(
        "--query",
        type=str,
        default="Can you explain Archimedes principle with an example?",
    )
    args = parser.parse_args()
    config = ProjectConfig()
    result = run_system_eval(config=config, query=args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
