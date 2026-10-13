"""Architecture tournament: pick the best candidate architecture without touching the test set.

    python -m fall_detection.tournament --data data/processed [--archs cnn1d,gru,tcn,transformer]

Each architecture is trained with the same recipe and scored by group-wise cross-validation on the
windows that are NOT in the frozen test split. Architectures that fail the deployment rules
(ONNX parity, latency, size) are disqualified. The winner is the best mean F1; only the winner is
then trained and judged once on the frozen test split by `fall_detection.promotion compare`.
Selecting on the test set instead would reward luck, not the architecture.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fall_detection import evaluation, promotion
from fall_detection.architectures import ARCHITECTURES, DEFAULT_ARCH, build_model


def cv_probabilities(X, y, groups, arch: str, seed: int, folds: int = 5) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(
        StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups)
    ):
        inner = StratifiedGroupKFold(6, shuffle=True, random_state=seed)
        fit, val = next(inner.split(X[tr], y[tr], groups[tr]))
        model = evaluation.train_one(
            X[tr][fit], y[tr][fit], X[tr][val], y[tr][val], seed + k, arch=arch
        )
        oof[te] = evaluation.predict(model, X[te])
    return oof


def static_rules(checks: dict) -> list[dict]:
    """Deployment rules that do not depend on the weights (an untrained model is enough)."""
    return [
        r
        for r in promotion.decide({"recall": 0}, {"recall": 0}, {"ci95": [1, 1]}, checks)[1]
        if not r["rule"].startswith(("F1", "recall"))
    ]


def run(X, y, groups, archs: list[str], seeds: int, baseline: str) -> dict:
    report = {"architectures": {}, "baseline": baseline}
    probs = {}
    for arch in archs:
        rules = static_rules(promotion.deployment_checks(build_model(arch)))
        runs = [cv_probabilities(X, y, groups, arch, 42 + s) for s in range(seeds)]
        metrics = [evaluation.confusion(y, (p > 0.5).astype(int)) for p in runs]
        probs[arch] = np.mean(runs, axis=0)
        report["architectures"][arch] = {
            "f1_mean": round(float(np.mean([m["f1"] for m in metrics])), 4),
            "recall_mean": round(float(np.mean([m["recall"] for m in metrics])), 4),
            "accuracy_mean": round(float(np.mean([m["accuracy"] for m in metrics])), 4),
            "f1_per_seed": [round(m["f1"], 4) for m in metrics],
            "deployment_rules": rules,
            "eligible": all(r["passed"] for r in rules),
        }
        print(arch, report["architectures"][arch]["f1_mean"], flush=True)

    for arch in archs:
        if arch != baseline and baseline in probs:
            report["architectures"][arch]["delta_f1_vs_baseline"] = promotion.paired_bootstrap_f1(
                y, probs[baseline], probs[arch], groups, n_boot=1000
            )
    eligible = {a: r for a, r in report["architectures"].items() if r["eligible"]}
    report["winner"] = max(eligible, key=lambda a: eligible[a]["f1_mean"]) if eligible else None
    return report


def to_markdown(report: dict) -> str:
    lines = [
        "# Architecture tournament",
        "",
        "Group-wise 5-fold CV on the windows outside the frozen test split.",
        "",
        "| Architecture | F1 | Recall | Accuracy | Delta F1 vs baseline (95% CI) | Eligible |",
        "|---|---|---|---|---|---|",
    ]
    for arch, r in report["architectures"].items():
        d = r.get("delta_f1_vs_baseline")
        delta = (
            f"{d['delta_f1_mean']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})" if d else "-"
        )
        lines.append(
            f"| {arch} | {r['f1_mean']:.3f} | {r['recall_mean']:.3f} | {r['accuracy_mean']:.3f} "
            f"| {delta} | {'yes' if r['eligible'] else 'no'} |"
        )
    lines += ["", f"**Winner: {report['winner']}** (baseline: {report['baseline']})"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--archs", default=",".join(ARCHITECTURES))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", default="reports/tournament")
    args = parser.parse_args()

    X, y, groups = promotion.load_data(args.data)
    split = json.loads(Path(promotion.SPLIT_PATH).read_text())
    promotion.check_split(split, X, y)
    keep = np.setdiff1d(np.arange(len(X)), split["test_indices"])  # never look at the test set
    baseline = json.loads(Path(promotion.CHAMPION_META).read_text()).get("arch", DEFAULT_ARCH)

    report = run(X[keep], y[keep], groups[keep], args.archs.split(","), args.seeds, baseline)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "tournament.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    Path(args.out, "tournament.md").write_text(md)
    print(md)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"winner={report['winner'] or ''}\n")


if __name__ == "__main__":
    main()
