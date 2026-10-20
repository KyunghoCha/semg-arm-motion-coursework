"""Last development round: conditional angle heads and amplitude features."""
import sys, time, json
import numpy as np
import torch
from torch import nn
import gpu
import phase2
from common import *

CANDIDATES = {
    "spectral256_hierarchical": "spectral256",
    "spectral256_hierarchical_rms": "spectral256_rms",
    "spectral256_gain_multitask": "spectral256",
}


class Model(nn.Module):
    def __init__(self, name, mean, scale):
        super().__init__()
        self.name = name
        self.multi = True
        if name == "spectral256_gain_multitask":
            self.base = phase2.Model(name, mean, scale)
            return
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.body = nn.Sequential(
            nn.Conv1d(645, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(.1),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(.1),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(.1))
        self.gesture = nn.Linear(128, 4)
        size = 133 if name.endswith("_rms") else 128
        self.angle = nn.Sequential(nn.Linear(size, 64), nn.LeakyReLU(.1),
                                   nn.Dropout(.25), nn.Linear(64, 8))

    def forward(self, x, aux=False):
        if self.name == "spectral256_gain_multitask":
            if self.training:
                gains = torch.empty(len(x), 5, 1, 1, device=x.device).uniform_(-.15, .15)
                gains += torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(-.15, .15)
                x = (x.reshape(len(x), 5, -1, x.shape[-1]) +
                     gains*(2/np.log(10))).reshape_as(x)
            return self.base(x, aux=aux)
        normal = (x-self.mean)/self.scale
        h = self.body(normal[:, :645])
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        g = self.gesture(h)
        angle_input = torch.cat([h, normal[:, 645:].mean(-1)], 1) if self.name.endswith("_rms") else h
        angle = self.angle(angle_input).reshape(-1, 4, 2).log_softmax(-1)
        joint = g.log_softmax(-1)[:, :, None] + angle
        marginal_angle = joint.logsumexp(1)
        z = joint.flatten(1)
        return (z, g, marginal_angle) if aux else z


def prepare():
    path = DATA/"spectral256_rms.npy"
    assert not path.exists()
    x = np.load(OLD/"data/raw1000.npy")
    rms = np.log10(np.maximum(np.sqrt(np.mean(x.astype("float64")**2, -1)), 1e-12))
    spectral = np.load(DATA/"spectral256.npy")
    arr = np.concatenate([spectral, np.repeat(rms[:, :, None], spectral.shape[-1], -1)], 1).astype("float32")
    assert np.isfinite(arr).all()
    np.save(path, arr)
    write(OUT/"results/PHASE3_FROZEN.json", {
        "timestamp": stamp(), "source_sha256": sha(__file__),
        "amendment_sha256": sha(HERE/"AMENDMENT_2.md"), "candidates": CANDIDATES,
        "inputs": {key: sha(DATA/f"{key}.npy") for key in set(CANDIDATES.values())}})
    print(path.name, arr.shape, flush=True)


def frozen():
    phase2.frozen()
    f = json.loads((OUT/"results/PHASE3_FROZEN.json").read_text())
    assert f["source_sha256"] == sha(__file__)
    assert f["amendment_sha256"] == sha(HERE/"AMENDMENT_2.md")
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
        for _ in range(30):
            opt.zero_grad()
            z, g, a = model(x, aux=True)
            loss = nn.functional.cross_entropy(z, y) + .3*nn.functional.cross_entropy(g, y//2) + .3*nn.functional.cross_entropy(a, y%2)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            z = model(x)
            assert torch.equal(z, model(x))
            accuracy = (z.argmax(1) == y).float().mean().item()
            assert accuracy >= .9, (name, accuracy)
            second = Model(name, np.zeros((1, channels, 1)), np.ones((1, channels, 1))).cuda()
            second.load_state_dict(model.state_dict()); second.eval()
            assert torch.equal(z, second(x))
            if "hierarchical" in name:
                assert torch.allclose(z.exp().sum(1), torch.ones(24, device="cuda"), atol=1e-6)
        checks[name] = {"inference_deterministic": True, "checkpoint_replay_exact": True,
                        "toy_accuracy": accuracy, "parameters": sum(p.numel() for p in model.parameters())}
    write(OUT/"reports/phase3_self_check.json", checks)
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
