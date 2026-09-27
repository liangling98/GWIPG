import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gwipg.benchmarks.graph_alignment import (
    add_edge_flip_noise,
    generate_gw_graph_instance_noisy,
    matching_accuracy,
    perm_to_ground_truth_matrix,
)
from gwipg.benchmarks.snare_seq import download_snare_data, scot_application_metrics
from gwipg.benchmarks.spheres import exact_sphere_distance


def test_vectorized_noise_retains_original_seed_sequence():
    A = np.zeros((12, 12))
    expected = A.copy()
    rng = np.random.default_rng(7)
    for i in range(12):
        for j in range(i + 1, 12):
            if rng.random() < 0.2:
                expected[i, j] = expected[j, i] = 1
    np.testing.assert_array_equal(add_edge_flip_noise(A, 0.2, 7), expected)


def test_graph_pairing_and_normalization():
    _, _, C1, C2, _, _, perm = generate_gw_graph_instance_noisy(
        n=20,
        edge_prob=0.2,
        noise_level=0,
        seed=42,
        normalize_cost=False,
    )
    np.testing.assert_array_equal(C2, C1[perm][:, perm])
    assert C1.max() > 1
    assert matching_accuracy(perm_to_ground_truth_matrix(perm) / 20, perm) == 1
    with pytest.raises(ValueError):
        generate_gw_graph_instance_noisy(n=20, edge_prob=0)


def test_snare_metrics_recognize_perfect_hidden_pairing():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 5))
    permutation = rng.permutation(12)
    inverse = np.argsort(permutation)
    labels = np.arange(12) % 3
    T = np.zeros((12, 12))
    T[np.arange(12), inverse] = 1 / 12
    result = scot_application_metrics(
        T, x, x[permutation], inverse, labels, labels[permutation], label_neighbors=1
    )
    assert result["foscttm"] == 0
    assert result["lta_x_to_y"] == 1
    assert result["lta_y_to_x"] == 1


def test_offline_data_mode_and_corrupt_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline"):
        download_snare_data(tmp_path, offline=True)
    (tmp_path / "snare_chromatin.txt").write_text("broken")
    with pytest.raises(ValueError, match="Checksum"):
        download_snare_data(tmp_path, offline=True)


def test_sphere_reference_and_loss_convention():
    distance, loss = exact_sphere_distance(1, 2)
    assert distance == pytest.approx(0.48232634142421155)
    assert (loss / 16) ** 0.25 == pytest.approx(distance)
    assert exact_sphere_distance(2, 2) == (0, 0)


@pytest.mark.parametrize(
    "module",
    [
        "graph_alignment",
        "snare_seq",
        "spheres",
        "inner_accuracy",
        "projection_cost",
        "gaussian",
        "lowrank",
        "paper",
    ],
)
def test_cli_help(module):
    result = subprocess.run(
        [sys.executable, "-m", f"gwipg.benchmarks.{module}", "--help"],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_archived_paper_results_have_not_changed():
    root = Path(__file__).resolve().parents[1] / "paper_results"
    for item in json.loads((root / "manifest.json").read_text()):
        assert hashlib.sha256((root / item["file"]).read_bytes()).hexdigest() == item["sha256"]


def test_graph_pipeline_writes_results_and_metadata(tmp_path):
    from gwipg.benchmarks.graph_alignment import main
    import pandas as pd

    main(["--smoke-test", "--output-dir", str(tmp_path)])
    raw = pd.read_csv(tmp_path / "graph_alignment_raw.csv")
    assert set(raw.method) == {"CG", "PPA", "PGD", "BAPG", "iPG"}
    assert set(raw.status) == {"success"}
    assert (tmp_path / "run_metadata.json").is_file()


def test_paper_runner_writes_sphere_results(tmp_path):
    from gwipg.benchmarks.paper import main
    import pandas as pd

    main(["spheres", "--quick", "--output-dir", str(tmp_path)])
    raw = pd.read_csv(tmp_path / "quick/spheres/smoke_test/gw_sphere_raw.csv")
    assert set(raw.status) == {"success"}
    assert np.isfinite(raw.loss).all()


def test_download_checksum_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    import io
    from gwipg.benchmarks import snare_seq

    monkeypatch.setattr(
        snare_seq.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt")
    )
    with pytest.raises(ValueError, match="checksum"):
        download_snare_data(tmp_path)
    assert list(tmp_path.iterdir()) == []
