"""Freeze all adaptive choices before opening outer/test results."""
import sys, time, json
import numpy as np
import torch
import gpu
import phase2
import phase3
from common import *

NEURAL = {**GPU, **phase2.CANDIDATES, **phase3.CANDIDATES}
SINGLE = [*CPU, *NEURAL]
ONE_SECOND = [n for n in NEURAL if n.startswith("spectral") and n != "spectral2s"]


def lock():
    phase3.frozen()
    target = OUT/"results/ADAPTIVE_SELECTION_LOCK.json"
    assert not target.exists(), "selection is already frozen"
    assert not list((OUT/"results/nested_outer").glob("**/summary.json"))
    assert not list((OUT/"results/final").glob("**/summary.json"))
    summaries = {n: summarize(OUT/f"results/screen/{n}") for n in SINGLE}
    best = sorted(SINGLE, key=lambda n: (-summaries[n]["macro_f1"], n))[0]
    top2 = sorted(ONE_SECOND, key=lambda n: (-summaries[n]["macro_f1"], n))[:2]
    folds = []
    for k in range(5):
        a, b = [np.load(OUT/f"results/screen/{n}/fold{k}/validation.npz") for n in top2]
        assert np.array_equal(a["indices"], b["indices"]) and np.array_equal(a["true"], b["true"])
        p = (a["probabilities"]+b["probabilities"])/2
        folds.append(score(a["true"], p.argmax(1)))
    ensemble = {"members": top2, "folds": folds,
                "accuracy": float(np.mean([r["accuracy"] for r in folds])),
                "macro_f1": float(np.mean([r["macro_f1"] for r in folds]))}
    selected = "ensemble" if ensemble["macro_f1"] > summaries[best]["macro_f1"] else best
    members = top2 if selected == "ensemble" else [best]
    evaluate = sorted(set(["spectral1s", "spectral_multitask", "legacy_svm1", best] + members))
    frozen = {
        "timestamp": stamp(), "best_single": best, "selected": selected,
        "selected_members": members, "evaluate": evaluate, "final_seeds": [42, 2026],
        "single_candidates": summaries, "ensemble_candidate": ensemble,
        "source_hashes": {p.name: sha(p) for p in HERE.glob("*.py")},
        "lock_hashes": {p.name: sha(p) for p in (OUT/"results").glob("*LOCK.json")},
        "split_sha256": sha(DATA/"splits.npz"),
        "no_further_tuning": True,
        "original_test_is_reused": True,
    }
    write(target, frozen)
    print("ADAPTIVE LOCK", selected, "best single", best, "members", members,
          "ensemble", ensemble["accuracy"], ensemble["macro_f1"], flush=True)


def checked_lock():
    phase3.frozen()
    p = OUT/"results/ADAPTIVE_SELECTION_LOCK.json"
    r = json.loads(p.read_text())
    for name, digest in r["source_hashes"].items():
        assert sha(HERE/name) == digest
    assert sha(DATA/"splits.npz") == r["split_sha256"]
    return r


def final():
    locked = checked_lock()
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    s = splits(); start = time.monotonic()
    assert set(s["recording"][s["development"]]).isdisjoint(s["recording"][s["test"]])
    for name in locked["evaluate"]:
        if name not in NEURAL:
            continue  # CPU final uses the original frozen trainer.
        gpu.Model = (phase3.Model if name in phase3.CANDIDATES else
                     phase2.Model if name in phase2.CANDIDATES else phase2.OriginalModel)
        array = np.load(DATA/f"{NEURAL[name]}.npy")
        x = torch.from_numpy(array).cuda()
        for seed in locked["final_seeds"]:
            folder = OUT/f"results/final/{name}/seed{seed}"
            if (folder/"summary.json").exists():
                continue
            epochs = locked["single_candidates"][name]["epochs"]
            r = gpu.fit(name, array, x, s["y"], s["development"], s["test"],
                        folder, seed, start, epochs=epochs)
            print("FINAL", name, seed, r["accuracy"], r["macro_f1"], flush=True)
        del x
        torch.cuda.empty_cache()


def verify():
    locked = checked_lock()
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    s = splits(); records = {}
    for name in locked["evaluate"]:
        if name not in NEURAL:
            continue
        cls = (phase3.Model if name in phase3.CANDIDATES else
               phase2.Model if name in phase2.CANDIDATES else phase2.OriginalModel)
        array = np.load(DATA/f"{NEURAL[name]}.npy")
        x = torch.from_numpy(array).cuda()
        expected_mean, expected_scale = gpu.old.normalization(array, s["development"])
        for seed in locked["final_seeds"]:
            folder = OUT/f"results/final/{name}/seed{seed}"
            model = cls(name, expected_mean, expected_scale).cuda()
            state = torch.load(folder/"last.pt", weights_only=True)
            prefix = "base." if name == "spectral256_gain_multitask" else ""
            assert np.array_equal(state[prefix+"mean"].cpu().numpy(), expected_mean)
            assert np.array_equal(state[prefix+"scale"].cpu().numpy(), expected_scale)
            model.load_state_dict(state)
            saved = np.load(folder/"validation.npz")
            pred = gpu.predict(model, x, saved["indices"])
            assert np.array_equal(pred, saved["probabilities"])
            model.eval()
            with torch.no_grad():
                sample = x[saved["indices"][:1]]
                for _ in range(10): model(sample)
                torch.cuda.synchronize(); t0 = time.monotonic()
                for _ in range(100): model(sample)
                torch.cuda.synchronize()
                ms = (time.monotonic()-t0)*10
            records[f"{name}/{seed}"] = {"checkpoint_replay_exact": True,
                "train_only_normalization_exact": True, "single_model_input_inference_ms": ms,
                "excludes_raw_loading_alignment_stft": True,
                "checkpoint_sha256": sha(folder/"last.pt")}
        del x
        torch.cuda.empty_cache()
    write(OUT/"reports/final_checkpoint_verification.json", records)
    print(records, flush=True)


if __name__ == "__main__":
    {"lock": lock, "final": final, "verify": verify}[sys.argv[1]]()
