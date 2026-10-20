"""Lock both the fixed nested procedure and the completed adaptive development choice."""
import json
import adaptive
from train import *


def main():
    adaptive.frozen()
    assert not list((OUT/"results/outer").glob("**/summary.json"))
    original = {n: summarize(OUT/f"results/screen/{n}") for n in CONFIG}
    extra = {n: summarize(OUT/f"results/adaptive/{n}") for n in adaptive.CANDIDATES}
    all_rows = {**original, **extra}
    rank = sorted(all_rows, key=lambda n: (-all_rows[n]["macro_f1"], n))
    one_second = [n for n in rank if n not in adaptive.CANDIDATES or adaptive.CANDIDATES[n]["context"] == "1s"]
    p = OUT/"results/DEVELOPMENT_SELECTION_LOCK.json"
    assert not p.exists()
    select()
    write(p, {"timestamp": stamp(), "best_development": rank[0], "best_1s": one_second[0],
              "rank": rank, "results": all_rows, "total_candidates": len(rank),
              "test_evaluation": False, "adaptive_frozen_sha256": sha(OUT/"results/ADAPTIVE_FROZEN.json"),
              "nested_selection_sha256": sha(OUT/"results/SELECTION_LOCK.json")})
    print("DEVELOPMENT LOCK", rank[0], all_rows[rank[0]]["accuracy"], all_rows[rank[0]]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
