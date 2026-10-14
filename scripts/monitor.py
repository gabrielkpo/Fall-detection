"""Monthly drift check.

python scripts/monitor.py --build-reference --data data/processed   # once, needs X.npy
python scripts/monitor.py                                            # CI / cron
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fall_detection import inference, monitoring

REFERENCE = "models/monitoring_reference.json"
CURRENT_LOG = "models/monitoring_current.jsonl"


def build_reference(data_dir: str, out: str) -> None:
    X = np.load(f"{data_dir}/X.npy").astype(np.float32)
    scores = inference.score_windows(inference.load_model(), X)
    meta = inference.load_metadata()
    payload = {
        "model_version": meta.get("version"),
        "n": len(X),
        "feature_names": monitoring.FEATURES,
        "features": monitoring.window_features(X).round(5).tolist(),
        "scores": scores.round(5).tolist(),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload))
    print(f"wrote {out} ({len(X)} windows)")


def to_markdown(report: dict) -> str:
    lines = [
        "# Monthly drift report",
        "",
        f"Model version: `{report['model_version']}`  ",
        (
            f"Monitor self-test: **{'passed' if report['self_test']['passed'] else 'FAILED'}** "
            "(quiet on a resample, fires on an injected shift)"
        ),
        "",
    ]
    cur = report.get("current")
    if cur is None:
        lines += [
            (
                "No production data this month (`models/monitoring_current.jsonl` absent). The browser demo "
                "sends nothing; only a self-hosted API with `FALL_LOG_PATH` produces logs."
            ),
        ]
    else:
        lines += [
            (
                f"Windows compared: {cur['n_current']} (reference {cur['n_reference']}). "
                f"Positive rate {cur['positive_rate_reference']:.2f} -> {cur['positive_rate_current']:.2f}"
            ),
            "",
            "| Feature | KS | p | PSI | Drift |",
            "|---|---|---|---|---|",
        ]
        for r in cur["features"]:
            lines.append(
                f"| {r['feature']} | {r['ks']} | {r['ks_p']} | {r['psi']} | "
                f"{'YES' if r['drift'] else 'no'} |"
            )
        lines += ["", f"**Drift detected: {'YES' if cur['drift'] else 'no'}**"]
    return "\n".join(lines) + "\n"


def run(out_dir: str) -> dict:
    ref = monitoring.load_reference(REFERENCE)
    report = {
        "model_version": ref.get("model_version"),
        "self_test": monitoring.self_test(ref["features"], ref["scores"]),
        "current": None,
    }
    if Path(CURRENT_LOG).exists():
        feats, scores = monitoring.read_log(CURRENT_LOG)
        if len(feats) >= 30:
            report["current"] = monitoring.drift_report(
                ref["features"], feats, ref["scores"], scores
            )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "drift.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    Path(out_dir, "drift.md").write_text(md)
    print(md)
    flag = bool(report["current"] and report["current"]["drift"])
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"drift={'true' if flag else 'false'}\n")
            f.write(f"selftest={'true' if report['self_test']['passed'] else 'false'}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    if args.build_reference:
        build_reference(args.data, REFERENCE)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
