import hashlib
import json
from pathlib import Path


def test_champion_metadata_matches_weights():
    """Drift detection between models/champion.json and the weights file."""
    meta = json.loads(Path("models/champion.json").read_text())
    digest = hashlib.sha256(Path(meta["weights_file"]).read_bytes()).hexdigest()
    assert digest == meta["weights_sha256"]


def test_frozen_split_is_consistent():
    split = json.loads(Path("models/frozen_split.json").read_text())
    assert len(split["test_indices"]) == len(set(split["test_indices"]))
    assert split["n_windows"] > max(split["test_indices"])
