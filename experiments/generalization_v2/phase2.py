"""Adaptive development only; the frozen primary family is unchanged."""
import sys, time, json
import numpy as np
import torch
from torch import nn
import gpu
from common import *

CANDIDATES = {
    "spectral_weighted": "spectral1s",
    "spectral128_multitask": "spectral128",
    "spectral256_multitask": "spectral256",
    "spectral_gain_multitask": "spectral1s",
}
OriginalModel = gpu.Model


class Model(OriginalModel):
    def __init__(self, name, mean, scale):
        super().__init__(name, mean, scale)
        if mean.shape[1] != 165:
            self.body[0] = nn.Conv1d(mean.shape[1], 64, 1)

    def forward(self, x, aux=False):
        if self.training and self.name == "spectral_gain_multitask":
            gains = torch.empty(len(x), 5, 1, 1, device=x.device).uniform_(-.15, .15)
            gains += torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(-.15, .15)
            x = (x.reshape(len(x), 5, -1, x.shape[-1]) +
                 gains * (2 / np.log(10))).reshape_as(x)
        return super().forward(x, aux)


def prepare():
    x = np.load(OLD/"data/raw1000.npy")
    for size, hop in [(128, 16), (256, 32)]:
        path = DATA/f"spectral{size}.npy"
        assert not path.exists()
        # Chunking keeps temporary float64 FFT allocations bounded.
        chunks = []
        for part in np.array_split(x, 16):
            frames = np.lib.stride_tricks.sliding_window_view(part, size, axis=-1)[..., ::hop, :]
            power = np.abs(np.fft.rfft(frames*np.hanning(size), axis=-1))**2
            chunks.append(np.log10(np.maximum(power, 1e-12)).transpose(0, 1, 3, 2)
                          .reshape(len(part), 5*(size//2+1), -1).astype("float32"))
        value = np.concatenate(chunks)
        assert np.isfinite(value).all()
        np.save(path, value)
        print(path.name, value.shape, flush=True)
    write(OUT/"results/PHASE2_FROZEN.json", {
        "timestamp": stamp(), "source_sha256": sha(__file__),
        "amendment_sha256": sha(HERE/"AMENDMENT_1.md"),
        "candidates": CANDIDATES,
        "inputs": {key: sha(DATA/f"{key}.npy") for key in set(CANDIDATES.values())},
    })


def frozen():
    freeze()
    f = json.loads((OUT/"results/PHASE2_FROZEN.json").read_text())
    assert f["source_sha256"] == sha(__file__)
    assert f["amendment_sha256"] == sha(HERE/"AMENDMENT_1.md")
    assert f["candidates"] == CANDIDATES
    for key, value in f["inputs"].items():
        assert sha(DATA/f"{key}.npy") == value


def self_check():
    checks = {}
    gpu.old.set_seed(7)
    for name, key in CANDIDATES.items():
        channels, length = np.load(DATA/f"{key}.npy", mmap_mode="r").shape[1:]
        model = Model(name, np.zeros((1, channels, 1)), np.ones((1, channels, 1))).cuda()
        x = torch.randn(24, channels, length, device="cuda")
        y = torch.arange(24, device="cuda") % 8
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(20):
            opt.zero_grad()
            z = model(x)
            loss = nn.functional.cross_entropy(z, y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model(x)
            assert torch.equal(z, model(x)), "augmentation leaked into inference"
            assert torch.isfinite(z).all()
            accuracy = (z.argmax(1) == y).float().mean().item()
            assert accuracy >= .9, (name, accuracy)
            state = model.state_dict()
            second = Model(name, state["mean"].cpu().numpy(), state["scale"].cpu().numpy()).cuda()
            second.load_state_dict(state); second.eval()
            assert torch.equal(z, second(x))
        checks[name] = {"inference_deterministic": True, "checkpoint_replay_exact": True,
                        "toy_accuracy": accuracy, "parameters": sum(p.numel() for p in model.parameters())}
    write(OUT/"reports/phase2_self_check.json", checks)
    print(checks, flush=True)


def run(stage):
    frozen()
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    gpu.Model = Model
    if stage == "self-check":
        return self_check()
    assert stage == "screen"
    s = splits(); start = time.monotonic()
    for name, key in CANDIDATES.items():
        array = np.load(DATA/f"{key}.npy")
        x = torch.from_numpy(array).cuda()
        for folder, tr, va, seed, epochs in cases("screen", name):
            if (folder/"summary.json").exists():
                continue
            r = gpu.fit(name, array, x, s["y"], tr, va, folder, seed, start)
            print(name, folder.name, r["accuracy"], r["macro_f1"], flush=True)
        del x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(sys.argv[1])
