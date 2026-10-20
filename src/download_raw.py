"""Recover only the 64 pinned CSVs, preserving prior archives and outputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "prior/corrected_v3/preprocessing/manifest.json").read_text())
    out = ROOT / "raw"
    out.mkdir(exist_ok=True)

    def recover(entry):
        path = out / Path(entry["path"]).name
        if not path.exists():
            url = ("https://raw.githubusercontent.com/TestUser-alt-dev/"
                   "sEMG-Hand-Gesture-Elbow-Angle-Dataset/" + manifest["commit"] + "/"
                   + urllib.parse.quote(entry["path"]))
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        with path.with_suffix(".part").open("wb") as f:
                            shutil.copyfileobj(response, f)
                    path.with_suffix(".part").replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert len(data) == entry["size"], path
        assert digest == entry["sha256"], path
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["sha"]
        print(json.dumps({"file": path.name, "bytes": len(data), "sha256": digest}), flush=True)
        return {"file": path.name, "bytes": len(data), "sha256": digest}

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(recover, manifest["files"]))
    (ROOT / "raw_manifest_verified.json").write_text(json.dumps(verified, indent=2))
    print("Verified all 64 source CSVs.", flush=True)


if __name__ == "__main__":
    main()
