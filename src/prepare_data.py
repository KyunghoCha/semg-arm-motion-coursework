"""Audit pinned source data and build raw inputs at exactly the old window locations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / "prior/corrected_v3"


def basic(w):
    delta = np.diff(w, axis=1)
    return np.concatenate([
        np.sqrt(np.mean(w * w, axis=1)),
        np.mean(np.abs(w), axis=1),
        np.abs(delta).sum(axis=1),
        ((w[:, :-1] * w[:, 1:] < 0) & (np.abs(delta) >= .01)).sum(axis=1),
        (delta[:, :-1] * delta[:, 1:] < 0).sum(axis=1),
    ], axis=1)


def main():
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    reference = dict(np.load(PRIOR / "preprocessing/peak_energy1s70.npz"))
    manifest = json.loads((PRIOR / "preprocessing/manifest.json").read_text())
    waveform, metadata, record_audit, trials = [], [], [], []
    max_error = 0.
    for entry in manifest["files"]:
        path = ROOT / "raw" / Path(entry["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        df = pd.read_csv(path)
        assert set(df.columns) == {f"bandpass {i}" for i in range(1, 6)} | {"time", "mark", "onset"}
        x = df[[f"bandpass {i}" for i in range(1, 6)]].to_numpy(dtype=np.float64)
        assert np.isfinite(x).all() and np.allclose(np.diff(df.time), .001, atol=1e-8)
        marks = np.flatnonzero(df.mark.to_numpy() != 0)
        assert len(marks) == 30 and np.all(np.diff(marks) >= 4500)
        subject, condition = path.stem.split("_")
        angle, gesture = condition.split(" ")
        y = ["fist", "open", "pro", "extor"].index(gesture) * 2 + int(angle == "60")
        record_audit.append({
            "recording": path.name, "subject": subject, "label": y,
            "samples": len(x), "first_mark": int(marks[0]),
            "marker_intervals": np.unique(np.diff(marks), return_counts=True)[0].tolist(),
            "onset_rows": np.flatnonzero(df.onset.to_numpy() != 0).tolist(),
            "channel_rms": np.sqrt(np.mean(x*x, axis=0)).tolist(),
            "abs_max": np.max(np.abs(x), axis=0).tolist(),
        })
        for j, start in enumerate(marks):
            end = int(marks[j+1]) if j < 29 else min(len(x), int(start)+5000)
            cycle = x[start:end]
            candidates = np.arange(500, 3501, 100)
            cumulative = np.r_[0., np.cumsum(np.sum(cycle*cycle, axis=1))]
            energy = cumulative[candidates+1000] - cumulative[candidates]
            offset = int(candidates[np.argmax(energy)])
            peak = int(start)+offset
            tid = f"{path.stem}|mark{j+1:02d}"
            rows = np.flatnonzero(reference["trial"] == tid)
            assert len(rows) == 9
            assert np.all(reference["peak_start"][rows] == peak)
            assert np.array_equal(reference["window_start"][rows], peak+np.arange(9)*100)
            assert np.all(reference["y"][rows] == y)
            assert peak+1000 <= end
            segment = x[peak:peak+1000]
            wins = np.stack([segment[k:k+200] for k in range(0, 801, 100)])
            features = basic(wins)
            max_error = max(max_error, float(np.abs(features-reference["X"][rows, :25]).max()))
            assert np.allclose(features, reference["X"][rows, :25], rtol=1e-10, atol=1e-10)
            waveform.append(segment.T.astype("float32"))
            metadata.append((tid, path.name, subject, y, peak))
            ch_energy = np.sum(segment*segment, axis=0)
            fractions = [float(np.sum(cycle[k:k+1000]**2) / np.sum(cycle**2))
                         for k in range(0, 5000, 1000)]
            trials.append({"trial": tid, "recording": path.name, "marker_start": int(start),
                           "marker_end_exclusive": end, "peak_start": peak, "offset_ms": offset,
                           "energy_fractions_by_second": fractions,
                           "log_rms": np.log10(np.maximum(np.sqrt(ch_energy/1000), 1e-12)).tolist(),
                           "dominant_channel_fraction": float(ch_energy.max()/ch_energy.sum())})
        print("audited", path.name, flush=True)

    seq = np.stack(waveform)
    assert seq.shape == (1920, 5, 1000)
    windows = np.stack([seq[:, :, k:k+200] for k in range(0, 801, 100)], axis=1)
    windows = windows.reshape(-1, 5, 200)
    trial_ids = np.array([m[0] for m in metadata])
    assert np.array_equal(np.repeat(trial_ids, 9), reference["trial"])
    np.save(out / "raw200.npy", windows)
    np.save(out / "raw1000.npy", seq)
    np.savez(out / "metadata.npz", **{k: reference[k] for k in
             ["y", "recording", "trial", "subject", "window", "window_start", "peak_start"]})
    split_report = {}
    for protocol, variant in [("recording", "peak_energy1s25"), ("trial", "peak_energy1s70")]:
        split = dict(np.load(PRIOR / f"results/{protocol}/{variant}/split.npz"))
        for key in ["trial", "recording", "y", "window_start"]:
            assert np.array_equal(split[key], reference[key])
        dev, test, fold = split["development"], split["test"], split["fold"]
        assert np.all(dev ^ test) and np.all(fold[test] == -1)
        assert set(reference["trial"][dev]).isdisjoint(reference["trial"][test])
        for tid in trial_ids:
            idx = reference["trial"] == tid
            assert len(set(dev[idx])) == 1 and len(set(fold[idx])) == 1
        for k in range(5):
            tr, va = dev & (fold != k), dev & (fold == k)
            assert set(reference["trial"][tr]).isdisjoint(reference["trial"][va])
            if protocol == "recording":
                assert set(reference["recording"][tr]).isdisjoint(reference["recording"][va])
        for record in np.unique(reference["recording"]):
            supports = [t for t in trials if t["recording"] == record]
            for prev, nxt in zip(supports, supports[1:]):
                assert prev["marker_end_exclusive"] <= nxt["marker_start"]
                assert prev["peak_start"]+1000 <= nxt["peak_start"]
        np.savez(out / f"{protocol}_split.npz", **split)
        split_report[protocol] = {
            "development_windows": int(dev.sum()), "test_windows": int(test.sum()),
            "development_trials": int(dev[::9].sum()), "test_trials": int(test[::9].sum()),
            "development_recordings": len(set(reference["recording"][dev])),
            "test_recordings": len(set(reference["recording"][test])),
            "subject_overlap": sorted(set(reference["subject"][dev]) & set(reference["subject"][test])),
            "development_class_window_counts": np.bincount(reference["y"][dev], minlength=8).tolist(),
            "test_class_window_counts": np.bincount(reference["y"][test], minlength=8).tolist(),
        }
    # Describe shift only in the A development records; do not tune from holdout diagnostics.
    split_a = np.load(out/"recording_split.npz")
    dev_records = set(reference["recording"][split_a["development"]])
    dev_trials = [t for t in trials if t["recording"] in dev_records]
    means = {r: np.mean([t["log_rms"] for t in dev_trials if t["recording"] == r], axis=0)
             for r in sorted(dev_records)}
    residual = np.array([np.array(t["log_rms"])-means[t["recording"]] for t in dev_trials])
    offsets = np.array([t["offset_ms"] for t in dev_trials])
    audit = {"records": 64, "trials": 1920, "channels": 5, "classes": 8, "subjects": 8,
             "recomputed_25_feature_max_abs_error": max_error,
             "splits": split_report,
             "a_development_only": {
                 "offset_ms_quantiles": np.quantile(offsets, [0, .25, .5, .75, 1]).tolist(),
                 "peak_second_counts": np.bincount(np.argmax(
                     np.array([t["energy_fractions_by_second"] for t in dev_trials]), axis=1),
                     minlength=5).tolist(),
                 "between_record_log_rms_sd": np.std(list(means.values()), axis=0).tolist(),
                 "within_record_log_rms_sd": np.std(residual, axis=0).tolist(),
                 "dominant_channel_over_80pct": float(np.mean(
                     [t["dominant_channel_fraction"] > .8 for t in dev_trials])),
             },
             "interpretation": "Record differences include classes, people, electrode conditions; this does not isolate a causal source.",
             "marker_semantics": "Not verified as physiological movement onset.",
             "source_filter_limit": "Upstream preprocessing across trial boundaries cannot be undone."}
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "record_audit.json").write_text(json.dumps(record_audit, indent=2))
    (out / "trial_audit.json").write_text(json.dumps(trials, indent=2))
    (out / "sha256.json").write_text(json.dumps({
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
        if p.suffix in [".npy", ".npz"]}, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
