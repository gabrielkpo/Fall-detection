"""Compare input representations with group-wise cross-validation (same protocol as docs/EVALUATION.md).

    python scripts/feature_experiment.py --data data/processed --seeds 3
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fall_detection import evaluation, features
from fall_detection.promotion import paired_bootstrap_f1


def cv_probabilities(Xf, y, groups, seed, channels):
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xf, y, groups)
    ):
        inner = StratifiedGroupKFold(6, shuffle=True, random_state=seed)
        fit, val = next(inner.split(Xf[tr], y[tr], groups[tr]))
        model = evaluation.train_one(
            Xf[tr][fit], y[tr][fit], Xf[tr][val], y[tr][val], seed + fold, channels=channels
        )
        oof[te] = evaluation.predict(model, Xf[te])
    return oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="docs/feature_experiment.json")
    args = parser.parse_args()

    X = np.load(f"{args.data}/X.npy").astype(np.float32)
    y = np.load(f"{args.data}/y.npy").astype(np.float32)
    groups = evaluation.reconstruct_groups(X, y)

    variants = {
        "raw132 (current)": X,
        "centered_scaled_xy (66)": features.centered_scaled(X),
        "derived (11)": features.window_features(X),
        "derived without absolute height (10)": features.window_features(X, False),
    }
    results, oofs = {}, {}
    for name, Xf in variants.items():
        runs = []
        for seed in range(args.seeds):
            oof = cv_probabilities(Xf, y, groups, 42 + seed, Xf.shape[2])
            runs.append(oof)
        metrics = [evaluation.confusion(y, (o > 0.5).astype(int)) for o in runs]
        oofs[name] = np.mean(runs, axis=0)  # average probability over seeds for paired tests
        results[name] = {
            k: [round(float(m[k]), 3) for m in metrics] for k in ("accuracy", "precision", "recall", "f1")
        }
        print(name, {k: round(float(np.mean(v)), 3) for k, v in results[name].items()}, flush=True)

    base = "raw132 (current)"
    for name in variants:
        if name == base:
            continue
        b = paired_bootstrap_f1(y, oofs[base], oofs[name], groups, n_boot=1000)
        results[name]["delta_f1_vs_raw"] = b
        print(name, "delta F1 vs raw:", round(b["delta_f1_mean"], 3), [round(v, 3) for v in b["ci95"]])
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
