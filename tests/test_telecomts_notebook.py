"""Execute the analysis notebook offline against a tiny generated CSV dataset."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from clauseguard.datasets.csv_export import export_analysis_csv


def test_analysis_notebook_runs_offline(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat", reason="Install the optional analysis extra")
    nbclient = pytest.importorskip("nbclient", reason="Install the optional analysis extra")
    pytest.importorskip("ipykernel", reason="Install the optional analysis extra")
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    data.mkdir()
    shutil.copyfile(root / "data/scenarios/scenarios.example.jsonl", data / "scenarios.jsonl")
    export_analysis_csv(data / "scenarios.jsonl")
    kernel = tmp_path / "kernels/test-python"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Test Python", "language": "python",
    }))
    monkeypatch.setenv("TELECOMTS_SCENARIO_DIR", str(data))
    monkeypatch.setenv("TELECOMTS_EDA_DIR", str(tmp_path / "figures"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    manager = KernelManager(kernel_name="test-python", kernel_spec_manager=KernelSpecManager(
        kernel_dirs=[str(kernel.parent)],
    ))
    notebook = nbformat.read(root / "notebooks/telecomts_analysis.ipynb", as_version=4)
    nbclient.NotebookClient(notebook, km=manager, timeout=120,
                           resources={"metadata": {"path": str(root)}}).execute()
    assert {path.stem for path in (tmp_path / "figures").glob("*.png")} == {
        "origin_and_categories", "lengths_and_cadence", "missing_values",
        "annotated_window", "illustrative_distributions", "illustrative_correlations",
        "representative_traces", "text_and_lineage_coverage",
    }
    assert all(cell.get("execution_count") is not None for cell in notebook.cells
               if cell.cell_type == "code")
