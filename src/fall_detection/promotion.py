"""Frozen test split, candidate training and the champion/candidate promotion rules.

    python -m fall_detection.promotion freeze   --data data/processed
    python -m fall_detection.promotion train    --data data/processed [--arch transformer]
    python -m fall_detection.promotion compare  --data data/processed [--apply]

The candidate can use any architecture of `fall_detection.architectures` (choose it with
`fall_detection.tournament`). It replaces the champion only if every rule in `decide` holds:
the lower bound of the 95 % paired bootstrap interval (resampling videos) of the F1 gain is above
zero on the frozen test, recall does not drop, the model exports to ONNX faithfully, and it stays
within the latency and size budgets of the browser demo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from fall_detection import evaluation, inference
from fall_detection.architectures import DEFAULT_ARCH, n_parameters

SPLIT_PATH = "models/frozen_split.json"
CHAMPION_META = "models/champion.json"
CHAMPION_WEIGHTS = "models/champion.pth"
CANDIDATE_WEIGHTS = "models/candidate.pth"
CANDIDATE_META = "models/candidate.json"

# Promotion budgets (browser demo: one window scored every ~150 ms, alongside pose estimation)
MAX_RECALL_DROP = 0.02
MAX_LATENCY_MS = 5.0  # ONNX Runtime, CPU, one window, p95
MAX_PARAMETERS = 200_000
MAX_ONNX_DIFF = 1e-4


def fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float32).tobytes())
    return h.hexdigest()


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(f"{data_dir}/X.npy").astype(np.float32)
    y = np.load(f"{data_dir}/y.npy").astype(np.float32)
    return X, y, evaluation.reconstruct_groups(X, y)


def freeze(X, y, groups, seed: int = 42) -> dict:
    """Hold out one fold of videos. Never used for training or model selection."""
    from sklearn.model_selection import StratifiedGroupKFold

    _, test_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups))
    return {
        "seed": seed,
        "dataset_sha256": fingerprint(X, y),
        "n_windows": len(X),
        "test_indices": [int(i) for i in test_idx],
        "test_groups": sorted({int(g) for g in groups[test_idx]}),
        "test_falls": int(y[test_idx].sum()),
    }


def check_split(split: dict, X, y) -> None:
    if split["dataset_sha256"] != fingerprint(X, y):
        raise SystemExit("dataset differs from the one the frozen split was made on")


def paired_bootstrap_f1(y, p_champion, p_candidate, groups, n_boot=2000, seed=0) -> dict:
    """Delta F1 (candidate - champion), resampling whole videos."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[g] for g in rng.choice(uniq, len(uniq))])
        f_a = evaluation.confusion(y[idx], (p_champion[idx] > 0.5).astype(int))["f1"]
        f_b = evaluation.confusion(y[idx], (p_candidate[idx] > 0.5).astype(int))["f1"]
        deltas.append(f_b - f_a)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_f1_mean": float(np.mean(deltas)), "ci95": [float(lo), float(hi)]}


def train_candidate(
    X, y, groups, split: dict, seed: int = 7, arch: str = DEFAULT_ARCH
) -> torch.nn.Module:
    from sklearn.model_selection import StratifiedGroupKFold

    train = np.setdiff1d(np.arange(len(X)), split["test_indices"])
    fit, val = next(
        StratifiedGroupKFold(6, shuffle=True, random_state=seed).split(
            X[train], y[train], groups[train]
        )
    )
    return evaluation.train_one(
        X[train][fit], y[train][fit], X[train][val], y[train][val], seed=seed, arch=arch
    )


def _sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decide(champion: dict, candidate: dict, boot: dict, checks: dict) -> tuple[bool, list[dict]]:
    """Every rule must pass. Returns the decision and each rule with its value."""
    rules = [
        {
            "rule": "F1 gain: lower bound of the 95% CI > 0",
            "value": round(boot["ci95"][0], 4),
            "passed": boot["ci95"][0] > 0,
        },
        {
            "rule": f"recall does not drop by more than {MAX_RECALL_DROP}",
            "value": round(candidate["recall"] - champion["recall"], 4),
            "passed": candidate["recall"] >= champion["recall"] - MAX_RECALL_DROP,
        },
        {
            "rule": f"ONNX export faithful (max diff < {MAX_ONNX_DIFF})",
            "value": checks.get("max_abs_diff"),
            "passed": checks.get("max_abs_diff") is not None
            and checks["max_abs_diff"] < MAX_ONNX_DIFF,
        },
        {
            "rule": f"latency p95 <= {MAX_LATENCY_MS} ms per window (ONNX, CPU)",
            "value": checks.get("latency_p95_ms"),
            "passed": checks.get("latency_p95_ms") is not None
            and checks["latency_p95_ms"] <= MAX_LATENCY_MS,
        },
        {
            "rule": f"parameters <= {MAX_PARAMETERS}",
            "value": checks.get("parameters"),
            "passed": checks.get("parameters", MAX_PARAMETERS + 1) <= MAX_PARAMETERS,
        },
    ]
    return all(r["passed"] for r in rules), rules


def deployment_checks(model: torch.nn.Module) -> dict:
    """ONNX parity and latency (skipped, hence failing the rules, if onnxruntime is missing)."""
    import tempfile

    checks = {"parameters": n_parameters(model)}
    try:
        from fall_detection.export import export_onnx

        with tempfile.TemporaryDirectory() as tmp:
            checks.update(export_onnx(model, f"{tmp}/candidate.onnx"))
    except ImportError as exc:
        checks["error"] = f"ONNX check unavailable: {exc}"
    return checks


def write_champion(
    weights_src: str, metrics: dict, split: dict, note: str, arch: str = DEFAULT_ARCH
) -> None:
    meta = json.loads(Path(CHAMPION_META).read_text())
    major, minor, _patch = (int(v) for v in meta["version"].split("."))
    Path(CHAMPION_WEIGHTS).write_bytes(Path(weights_src).read_bytes())
    model = inference.load_model(CHAMPION_WEIGHTS, arch=arch)
    meta.update(
        {
            "version": f"{major}.{minor + 1}.0",
            "arch": arch,
            "architecture": f"{arch}, {n_parameters(model)} parameters",
            "weights_sha256": _sha256_file(CHAMPION_WEIGHTS),
            "trained": dt.date.today().isoformat() + ", fall_detection.promotion, windows of the "
            "frozen test split excluded",
            "evaluation": {
                "protocol": "frozen group-wise test split (models/frozen_split.json)",
                "n_test_windows": metrics["n"],
                "accuracy": round(metrics["accuracy"], 3),
                "precision": round(metrics["precision"], 3),
                "recall": round(metrics["recall"], 3),
                "f1": round(metrics["f1"], 3),
                "caveat": note,
            },
            "dataset_sha256": split["dataset_sha256"],
            "git_sha": os.environ.get("GITHUB_SHA", "local"),
        }
    )
    Path(CHAMPION_META).write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "train", "compare", "baseline"])
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="architecture of the candidate")
    parser.add_argument("--apply", action="store_true", help="write champion files if promoted")
    parser.add_argument("--report", default="reports/promotion.json")
    args = parser.parse_args()

    X, y, groups = load_data(args.data)

    if args.command == "freeze":
        split = freeze(X, y, groups)
        Path(SPLIT_PATH).write_text(json.dumps(split, indent=2) + "\n")
        print(f"froze {len(split['test_indices'])} windows in {len(split['test_groups'])} videos")
        return

    split = json.loads(Path(SPLIT_PATH).read_text())
    check_split(split, X, y)
    test = np.array(split["test_indices"])

    if args.command in ("train", "baseline"):
        model = train_candidate(X, y, groups, split, arch=args.arch)
        torch.save(model.state_dict(), CANDIDATE_WEIGHTS)
        Path(CANDIDATE_META).write_text(json.dumps({"arch": args.arch}) + "\n")
        print(f"wrote {CANDIDATE_WEIGHTS} ({args.arch})")

    if args.command == "train":
        return

    cand_arch = json.loads(Path(CANDIDATE_META).read_text())["arch"]
    candidate = inference.load_model(CANDIDATE_WEIGHTS, arch=cand_arch)
    cand_p = inference.score_windows(candidate, X[test])
    cand_m = evaluation.confusion(y[test], (cand_p > 0.5).astype(int))

    if args.command == "baseline":
        # The first champion was trained on a random split that included these windows, so it
        # cannot be compared fairly: the clean candidate becomes the new reference.
        write_champion(
            CANDIDATE_WEIGHTS,
            cand_m,
            split,
            "Baseline retrained without the frozen test windows; small test set, wide intervals.",
            arch=cand_arch,
        )
        print(json.dumps(cand_m, indent=2))
        return

    champ_arch = inference.load_metadata(CHAMPION_META).get("arch", DEFAULT_ARCH)
    champion = inference.load_model(CHAMPION_WEIGHTS, arch=champ_arch)
    champ_p = inference.score_windows(champion, X[test])
    champ_m = evaluation.confusion(y[test], (champ_p > 0.5).astype(int))
    boot = paired_bootstrap_f1(y[test], champ_p, cand_p, groups[test])
    promote, rules = decide(champ_m, cand_m, boot, deployment_checks(candidate))
    report = {
        "champion": {"arch": champ_arch, **champ_m},
        "candidate": {"arch": cand_arch, **cand_m},
        "paired_bootstrap": boot,
        "rules": rules,
        "promote": bool(promote),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if promote and args.apply:
        write_champion(CANDIDATE_WEIGHTS, cand_m, split, "Promoted: all rules passed.", cand_arch)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"promote={'true' if promote else 'false'}\n")
            f.write(f"arch={cand_arch}\n")


if __name__ == "__main__":
    main()
