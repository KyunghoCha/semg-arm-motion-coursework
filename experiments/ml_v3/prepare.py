"""Sample-local ML features; only the original A development rows are retained."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[2]
OLD = TASK/"step2_gpu_pc"
V2 = TASK/"step2_improvement_v2"
OUT = TASK/"step2_ml_v3"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covlog(x, normalize=False):
    x = x-x.mean(-1, keepdims=True)
    cov = x @ x.transpose(0, 2, 1)/x.shape[-1]
    trace = np.trace(cov, axis1=1, axis2=2).clip(1e-20)
    if normalize:
        cov = cov/trace[:, None, None]
        trace = np.ones(len(cov))
    cov = .95*cov + .05*trace[:, None, None]/5*np.eye(5)
    eig, vec = np.linalg.eigh(cov)
    logged = (vec*np.log(eig.clip(1e-20))[:, None, :]) @ vec.transpose(0, 2, 1)
    i, j = np.triu_indices(5)
    return logged[:, i, j]*np.where(i == j, 1, np.sqrt(2))


def correlations(x):
    x = x-x.mean(-1, keepdims=True)
    x = x/np.sqrt(np.sum(x*x, -1, keepdims=True)).clip(1e-20)
    corr = x@x.transpose(0, 2, 1)
    i, j = np.triu_indices(5, 1)
    return corr[:, i, j]


def main():
    data = OUT/"data"
    data.mkdir(parents=True, exist_ok=True)
    assert not (data/"manifest.json").exists(), "prepared data already exists"
    s = dict(np.load(V2/"data/splits.npz"))
    keep = s["development"]
    source_indices = np.flatnonzero(keep)
    assert not np.any(s["test"][source_indices])
    x = np.load(OLD/"data/raw1000.npy")[keep].astype("float64")
    legacy = np.load(V2/"data/cpu_legacy.npy")[keep]
    cached = np.load(OLD/"prior/corrected_v3/preprocessing/peak_energy1s70.npz")["X"].copy()
    cached[:, :15] = np.log10(np.maximum(cached[:, :15], 1e-12))
    cached = cached.reshape(-1, 9, 70)[keep]
    assert np.array_equal(legacy, np.c_[cached.mean(1), cached.std(1)])
    robust = np.c_[cached.mean(1), cached.std(1), np.median(cached, axis=1),
                   np.quantile(cached, .75, axis=1)-np.quantile(cached, .25, axis=1)]
    freq, power = signal.welch(x, fs=1000, window=np.hanning(256), nperseg=256,
                               noverlap=128, detrend="constant", axis=-1)
    bands = [(20, 50), (50, 80), (80, 110), (110, 150), (150, 200), (200, 260), (260, 330), (330, 451)]
    bp = np.stack([power[:, :, (freq >= lo)&(freq < hi)].sum(-1) for lo, hi in bands], -1)
    relative = bp/bp.sum(-1, keepdims=True).clip(1e-20)
    use = (freq >= 20)&(freq <= 450)
    spectrum = power[:, :, use]
    total = spectrum.sum(-1, keepdims=True).clip(1e-20)
    prob = spectrum/total
    mnf = np.sum(prob*freq[use], -1)
    mdf = freq[use][np.argmax(np.cumsum(prob, -1) >= .5, -1)]
    entropy = -np.sum(prob*np.log(prob.clip(1e-20)), -1)/np.log(use.sum())
    flatness = np.exp(np.log(spectrum.clip(1e-20)).mean(-1))/spectrum.mean(-1).clip(1e-20)
    spectral_relative = np.c_[np.log10(relative.clip(1e-12)).reshape(len(x), -1), mnf, mdf, entropy, flatness]
    rms = np.sqrt(np.mean(x*x, -1))
    logrms = np.log10(rms.clip(1e-12))
    spectral = np.c_[spectral_relative, np.log10(bp.clip(1e-20)).reshape(len(x), -1), logrms]
    i, j = np.triu_indices(5, 1)
    amplitude = np.c_[logrms-logrms.mean(1, keepdims=True), logrms[:, i]-logrms[:, j]]
    envelope = np.sqrt(np.mean(x.reshape(-1, 5, 10, 100)**2, -1))
    corr = np.c_[correlations(x), correlations(envelope)]
    smallbands = [(20, 80), (80, 150), (150, 250), (250, 451)]
    coh = []
    for a, b in zip(i, j):
        f, c = signal.coherence(x[:, a], x[:, b], fs=1000, window=np.hanning(256),
                                nperseg=256, noverlap=128, detrend="constant", axis=-1)
        coh.extend([c[:, (f >= lo)&(f < hi)].mean(-1) for lo, hi in smallbands])
    coherence = np.stack(coh, -1)
    fft = np.fft.rfft(x, axis=-1)
    freq1 = np.fft.rfftfreq(1000, .001)
    absolute_cov = [covlog(x)]; relative_cov = [covlog(x, True)]
    for lo, hi in smallbands:
        bandx = np.fft.irfft(fft*((freq1 >= lo)&(freq1 < hi)), n=1000, axis=-1)
        absolute_cov.append(covlog(bandx)); relative_cov.append(covlog(bandx, True))
    relations = np.c_[*absolute_cov, amplitude, corr, coherence]
    relative_relation = np.c_[*relative_cov, amplitude, corr, coherence]
    relative_legacy = np.delete(legacy, np.r_[0:15, 70:85], axis=1)
    uppercols = [k for k in range(140) if (k % 70) % 5 in (3, 4)]
    forearmcols = [k for k in range(140) if (k % 70) % 5 in (0, 1, 2)]
    arrays = {"legacy": legacy, "robust": robust, "spectrum": spectral, "relations": relations,
              "fusion": np.c_[legacy, spectral, relations],
              "relative": np.c_[relative_legacy, spectral_relative, relative_relation],
              "upper": legacy[:, uppercols], "forearm": legacy[:, forearmcols]}
    for key, arr in arrays.items():
        assert len(arr) == 1530 and np.isfinite(arr).all()
        np.save(data/f"{key}.npy", arr)
    split = {k: s[k][keep] for k in ["y", "recording", "subject", "trial", "fold"] + [f"inner{k}" for k in range(5)]}
    split["source_indices"] = source_indices
    np.savez(data/"splits.npz", **split)
    manifest = {"rows": 1530, "records": 51, "original_test_rows": 0,
                "features": {k: list(a.shape) for k, a in arrays.items()},
                "hashes": {p.name: sha(p) for p in data.iterdir()},
                "source_raw_sha256": sha(OLD/"data/raw1000.npy"),
                "source_split_sha256": sha(V2/"data/splits.npz"),
                "source_sha256": sha(__file__), "protocol_sha256": sha(HERE/"PROTOCOL.md")}
    (data/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest["features"], indent=2), flush=True)


if __name__ == "__main__":
    main()
