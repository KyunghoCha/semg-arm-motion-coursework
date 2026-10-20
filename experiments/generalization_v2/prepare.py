"""Build sample-local representations and freeze grouping; no training."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[2]
OLD = TASK / "step2_gpu_pc"
OUT = TASK / "step2_improvement_v2"
sys.path.insert(0, str(OLD))
from prepare_data import basic


def spectral(x):
    frames = np.lib.stride_tricks.sliding_window_view(x, 64, axis=-1)[..., ::16, :]
    power = np.abs(np.fft.rfft(frames*np.hanning(64), axis=-1))**2
    return np.log10(np.maximum(power,1e-12)).transpose(0,1,3,2).reshape(len(x),165,-1).astype("float32")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False)+"\n")


def grouped_folds(records, y, eligible, n, seed):
    unique = np.unique(records[eligible])
    labels = np.array([y[np.flatnonzero(records == r)[0]] for r in unique])
    fold = np.full(len(y), -1, dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(n, shuffle=True, random_state=seed).split(unique, labels)):
        fold[eligible & np.isin(records, unique[va])] = k
    assert np.all(fold[eligible] >= 0)
    for r in unique:
        assert len(np.unique(fold[records == r])) == 1
    return fold


def main():
    data = OUT/"data"
    meta = np.load(OLD/"data/metadata.npz")
    y, rec, subject, trial = [meta[k][::9] for k in ["y", "recording", "subject", "trial"]]
    a = np.load(OLD/"data/recording_split.npz")
    dev, test, fold = [a[k][::9] for k in ["development", "test", "fold"]]
    manifest = json.loads((OLD/"prior/corrected_v3/preprocessing/manifest.json").read_text())
    hashes = {Path(e["path"]).name: e["sha256"] for e in manifest["files"]}
    full, peak2, lecture, rows = [], [], [], []
    for name in dict.fromkeys(rec):
        path = OLD/"raw"/name
        assert digest(path) == hashes[name]
        df = pd.read_csv(path)
        signal = df[[f"bandpass {c}" for c in range(1, 6)]].to_numpy()
        marks = np.flatnonzero(df.mark.to_numpy() != 0)
        assert len(marks) == 30
        for i, mark in enumerate(marks):
            end = marks[i+1] if i < 29 else len(signal)
            assert end-mark >= 4999
            x = signal[mark:min(mark+5000,end)]
            padded = len(x) < 5000
            if padded:
                x = np.pad(x,((0,5000-len(x)),(0,0)),mode="edge")
            energy = np.r_[0., np.cumsum(np.sum(x*x, axis=1))]
            starts = np.arange(500, 2501, 100)
            offset = int(starts[np.argmax(energy[starts+2000]-energy[starts])])
            full.append(x.T.astype("float32"))
            peak2.append(x[offset:offset+2000].T.astype("float32"))
            lecture.append(basic(np.stack([x[k:k+200] for k in range(1500, 3301, 100)])))
            rows.append({"recording":name, "trial":f"{Path(name).stem}|mark{i+1:02d}",
                         "marker":int(mark), "end":int(end), "peak2_offset":offset,
                         "last_sample_repeated":padded})
    assert [r["trial"] for r in rows] == trial.tolist()
    full = np.stack(full)
    peak2 = np.stack(peak2)
    peak1 = np.load(OLD/"data/raw1000.npy")
    env = np.log10(np.maximum(np.sqrt(np.mean(full.reshape(-1,5,50,100).astype("float64")**2, axis=-1)),1e-12))
    relative = env-np.quantile(env,.2,axis=-1,keepdims=True)
    for name, arr in {"envelope_abs":env,"envelope_relative":relative,
                      "envelope_hybrid":np.concatenate([env,relative],axis=1),
                      "spectral1s":spectral(peak1),"spectral2s":spectral(peak2)}.items():
        assert np.isfinite(arr).all()
        np.save(data/f"{name}.npy", arr.astype("float32"))
    np.save(data/"lecture25.npy", np.stack(lecture))
    # Normalized spectrum shape and absolute amplitude are separated explicitly.
    power=np.abs(np.fft.rfft(peak1.astype("float64")*np.hanning(1000),axis=-1))**2
    freq=np.fft.rfftfreq(1000,.001)
    bands=np.stack([power[:,:,(freq>=lo)&(freq<hi)].sum(-1)
                    for lo,hi in zip([20,50,80,110,150,200,260,330],[50,80,110,150,200,260,330,451])],-1)
    shape=np.log10(np.maximum(bands/bands.sum(-1,keepdims=True),1e-12)).reshape(len(y),-1)
    rms=np.log10(np.maximum(np.sqrt(np.mean(peak1.astype("float64")**2,-1)),1e-12))
    corr=np.stack([np.corrcoef(x)[np.triu_indices(5,1)] for x in peak1])
    shape=np.c_[shape,rms,corr]
    legacy=np.load(OLD/"prior/corrected_v3/preprocessing/peak_energy1s70.npz")["X"].copy()
    legacy[:,:15]=np.log10(np.maximum(legacy[:,:15],1e-12))
    legacy=legacy.reshape(-1,9,70)
    feature2=[]
    for x in peak2:
        b=basic(np.stack([x[:,k:k+200].T for k in range(0,1801,100)]))
        b[:,:15]=np.log10(np.maximum(b[:,:15],1e-12))
        feature2.append(np.r_[b.mean(0),b.std(0)])
    for name,arr in {"legacy":np.c_[legacy.mean(1),legacy.std(1)],
                     "shape":shape,"env":env.reshape(len(y),-1),
                     "rel":relative.reshape(len(y),-1),"peak2":np.stack(feature2)}.items():
        assert np.isfinite(arr).all()
        np.save(data/f"cpu_{name}.npy",arr)
    split={"y":y,"recording":rec,"subject":subject,"trial":trial,
           "development":dev,"test":test,"fold":fold}
    for k in range(5):
        split[f"inner{k}"]=grouped_folds(rec,y,dev&(fold!=k),3,20261021+k)
        assert set(rec[dev&(fold==k)]).isdisjoint(rec[dev&(fold!=k)])
    ctrain,ctest=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    cfold=np.full(len(y),-1,int)
    for k,(_,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y)):
        cfold[va]=k
    split.update(c_train=ctrain,c_test=ctest,c_fold=cfold)
    np.savez(data/"splits.npz",**split)
    supports=[]
    for k in range(5):
        tr=dev&(fold!=k); va=dev&(fold==k)
        supports.append({"fold":k,"train_recordings_per_class":[len(set(rec[tr&(y==c)])) for c in range(8)],
                         "validation_records":len(set(rec[va]))})
    save(OUT/"reports/preparation.json",{"records":64,"trials":len(y),"A_dev_records":len(set(rec[dev])),
         "A_test_records":len(set(rec[test])),"fold_support":supports,
         "source_hashes_verified":64,"marker_cycles_disjoint":True,
         "original_raw1s_input":"old verified 1s array reused without modification",
         "class_subject_record_confounding":"one recording per participant x class; holding a record removes that participant-class from training",
         "preparation_code_sha256":digest(__file__),"protocol_sha256":digest(HERE/"PROTOCOL.md")})
    save(data/"source_metadata.json",rows)
    save(data/"sha256.json",{p.name:digest(p) for p in data.iterdir() if p.suffix in [".npy",".npz"]})
    print("PREPARATION COMPLETE", full.shape, supports, flush=True)


if __name__ == "__main__":
    main()
