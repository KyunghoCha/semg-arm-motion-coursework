"""Recreate the historical 70-feature cache locally; no signal data is redistributed."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from prepare_data import basic

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / "prior/corrected_v3"


def expanded(w):
    # Preserve the historical feature formulas and their column order.
    base = basic(w)
    centered = w - w.mean(1, keepdims=True)
    freq = np.fft.rfftfreq(200, d=.001)
    power = np.abs(np.fft.rfft(centered * np.hanning(200)[None, :, None], axis=1)) ** 2
    use = (freq >= 20) & (freq <= 450)
    p, f = power[:, use], freq[use]
    total = p.sum(1) + 1e-30
    mean_frequency = (p * f[None, :, None]).sum(1) / total
    median_frequency = f[np.argmax(np.cumsum(p, axis=1) >= total[:, None, :] * .5, axis=1)]
    bands = [power[:, (freq >= lo) & (freq < hi)].sum(1) / total
             for lo, hi in [(20, 60), (60, 120), (120, 250), (250, 451)]]
    variance = np.var(centered, axis=1) + 1e-30
    delta = np.diff(centered, axis=1)
    variance1 = np.var(delta, axis=1) + 1e-30
    variance2 = np.var(np.diff(delta, axis=1), axis=1) + 1e-30
    mobility = np.sqrt(variance1 / variance)
    complexity = np.sqrt(variance2 / variance1) / mobility
    log_rms = np.log(np.maximum(base[:, :5], 1e-12))
    ratios = log_rms - log_rms.mean(1, keepdims=True)
    return np.concatenate([base, mean_frequency, median_frequency, *bands,
                           mobility, complexity, ratios], axis=1)


def main():
    target = PRIOR / "preprocessing/peak_energy1s70.npz"
    if target.exists():
        print(f"Preserved existing feature cache: {target}. Use a new workspace to rebuild.")
        return
    manifest = json.loads((PRIOR / "preprocessing/manifest.json").read_text())
    parts = {k: [] for k in ["X", "y", "recording", "trial", "subject",
                             "window_start", "window", "peak_start"]}
    candidates = np.arange(500, 3501, 100)
    for entry in manifest["files"]:
        path = ROOT / "raw" / Path(entry["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path.name
        frame = pd.read_csv(path)
        signal = frame[[f"bandpass {i}" for i in range(1, 6)]].to_numpy(dtype=np.float64)
        assert np.isfinite(signal).all()
        marks = np.flatnonzero(frame.mark.to_numpy() != 0)
        assert len(marks) == 30 and np.all(np.diff(marks) >= 4500)
        subject, condition = path.stem.split("_")
        angle, gesture = condition.split(" ")
        label = ["fist", "open", "pro", "extor"].index(gesture) * 2 + int(angle == "60")
        for j, start in enumerate(marks):
            end = int(marks[j + 1]) if j < 29 else min(len(signal), int(start) + 5000)
            cycle = signal[start:end]
            assert len(cycle) >= 4500
            cumulative = np.r_[0., np.cumsum(np.sum(cycle * cycle, axis=1))]
            offset = int(candidates[np.argmax(cumulative[candidates + 1000] - cumulative[candidates])])
            peak = int(start) + offset
            segment = signal[peak:peak + 1000]
            windows = np.stack([segment[k:k + 200] for k in range(0, 801, 100)])
            parts["X"].append(expanded(windows))
            for key, values in {
                "y": [label] * 9, "recording": [path.name] * 9,
                "trial": [f"{path.stem}|mark{j + 1:02d}"] * 9,
                "subject": [subject] * 9, "window": list(range(9)),
                "window_start": [peak + k * 100 for k in range(9)], "peak_start": [peak] * 9,
            }.items():
                parts[key].extend(values)
        print(f"rebuilt {path.name}", flush=True)
    arrays = {k: np.concatenate(v) if k == "X" else np.array(v) for k, v in parts.items()}
    names = ["RMS", "MAV", "WL", "ZC", "SSC", "MNF", "MDF", "BP20_60",
             "BP60_120", "BP120_250", "BP250_450", "HjorthMobility", "HjorthComplexity", "LogRMSratio"]
    arrays["feature_names"] = np.array([f"{name}_CH{ch}" for name in names for ch in range(1, 6)])
    assert arrays["X"].shape == (17280, 70) and np.isfinite(arrays["X"]).all()
    assert len(set(arrays["trial"])) == 1920
    # The published split metadata fixes labels, ordering and temporal support.
    for protocol, variant in [("recording", "peak_energy1s25"), ("trial", "peak_energy1s70")]:
        with np.load(PRIOR / f"results/{protocol}/{variant}/split.npz") as split:
            for key in ["y", "recording", "trial", "window_start"]:
                assert np.array_equal(arrays[key], split[key]), (protocol, key)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target)
    print(f"Saved {target.name}; shape={arrays['X'].shape}; numpy={np.__version__}; pandas={pd.__version__}")


if __name__ == "__main__":
    main()
