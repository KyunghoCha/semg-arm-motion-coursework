"""Run an experiment in its own directory, leaving the reviewed results unchanged."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
ACTIONS = {
    "download": ("download_raw.py", []),
    "features": ("rebuild_reference_features.py", []),
    "prepare": ("prepare_data.py", []),
    "self-check": ("train_gpu.py", ["self-check"]),
    "gpu-cv": ("train_gpu.py", ["cv"]),
    "gpu-extension-cv": ("train_extension.py", ["cv"]),
    "cpu-cv": ("cpu_trial_control.py", ["cv"]),
    "gpu-final": ("train_extension.py", ["final"]),
    "cpu-final": ("cpu_trial_control.py", ["final"]),
    "report": ("report_results.py", []),
    "verify-checkpoints": ("verify_checkpoints.py", []),
}


def copy_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(destination.read_bytes()).digest(), (
            f"Existing file differs: {destination}. Choose a new --workspace; existing work was preserved.")
    else:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--workspace", type=Path, default=PROJECT/"runs/reproduction")
    args = parser.parse_args()
    work = args.workspace.resolve()
    work.mkdir(parents=True, exist_ok=True)
    for source in (PROJECT/"src").glob("*.py"):
        copy_once(source, work/source.name)
    for name in ["PROTOCOL_GPU.md", "PROTOCOL_EXTENSION.md"]:
        copy_once(PROJECT/"configs"/name, work/name)
    for source in (PROJECT/"reference").rglob("*"):
        if source.is_file():
            copy_once(source, work/"prior"/source.relative_to(PROJECT/"reference"))
    script, rest = ACTIONS[args.action]
    print(f"Experiment workspace: {work}", flush=True)
    subprocess.run([sys.executable, str(work/script), *rest], check=True)


if __name__ == "__main__":
    main()
