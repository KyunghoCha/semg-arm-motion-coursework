"""Read-only replay of the locked final checkpoints; no fitting or selection."""
import json
import numpy as np
import torch
import train_extension as ext

base = ext.base
ROOT = base.ROOT


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    checks = []
    for protocol in ["recording", "trial"]:
        split = dict(np.load(ROOT/f"data/{protocol}_split.npz"))
        selection = json.loads((ROOT/f"results/{protocol}/FINAL_SELECTION_LOCK.json").read_text())
        for unit, choice in selection["selected"].items():
            candidate = choice["candidate"]
            step = 9 if unit == "1000" else 1
            dev, test = split["development"][::step], split["test"][::step]
            x_np = ext.load_input(candidate)
            x = torch.from_numpy(x_np).cuda()
            mean, scale = base.normalization(x_np, dev)
            folder = ROOT/f"results/{protocol}/{candidate}/final"
            saved = np.load(folder/"predictions.npz")
            for i, seed in enumerate(base.CONFIG["final_seeds"]):
                model = base.restore(folder/f"seed{seed}/last.pt")
                assert np.array_equal(mean, model.mean.cpu().numpy())
                assert np.array_equal(scale, model.scale.cpu().numpy())
                p = base.predict(model, x, np.flatnonzero(test))
                error = float(np.max(np.abs(p-saved["seed_probabilities"][i])))
                assert error <= 1e-7, (protocol, candidate, seed, error)
                checks.append({"protocol": protocol, "candidate": candidate, "seed": seed,
                               "normalization_uses_development_only": True,
                               "max_probability_replay_error": error})
                del model
            del x, x_np
            torch.cuda.empty_cache()
    base.save_json(ROOT/"report/checkpoint_replay.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
