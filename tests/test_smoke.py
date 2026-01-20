import os
import subprocess
from pathlib import Path


def test_demo_pipeline_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    outdir = tmp_path / "demo_run"

    cmd = [
        "python",
        str(repo_root / "src" / "explain_omics_pipeline.py"),
        "--inputs",
        str(repo_root / "data" / "example_inputs"),
        "--outdir",
        str(outdir),
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))

    csv_dir = outdir / "explain_omics" / "csv"
    assert (csv_dir / "Triplets_HostModule_Pathway_Microbe.csv").exists()
    assert (csv_dir / "Microbe_to_Pathway_Links.csv").exists()
    assert (csv_dir / "Pathway_to_HostModule_Links.csv").exists()
