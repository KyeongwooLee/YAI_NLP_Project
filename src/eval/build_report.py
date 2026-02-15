from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config: ProjectConfig) -> dict:
    ensure_project_dirs(config)
    correctness = _read_json(config.reports_dir / "correctness_eval.json")
    personalization = _read_json(config.reports_dir / "personalization_eval.json")
    system = _read_json(config.reports_dir / "system_eval.json")
    continual = _read_json(config.logs_dir / "continual_update_summary.json")
    latest_inference = _read_json(config.logs_dir / "latest_inference.json")

    report = {
        "correctness": correctness,
        "personalization": personalization,
        "system": system,
        "continual_update": continual,
        "latest_inference": latest_inference,
    }
    (config.reports_dir / "final_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final merged report.")
    _ = parser.parse_args()
    config = ProjectConfig()
    report = build_report(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
