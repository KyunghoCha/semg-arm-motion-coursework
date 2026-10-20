"""Shared immutable splits, score definitions and selection gates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
TASK=HERE.parents[2]
OLD=TASK/"step2_gpu_pc"
OUT=TASK/"step2_improvement_v2"
DATA=OUT/"data"
CPU={"legacy_svm1":("legacy",1), "shape_svm1":("shape",1),
     "shape_svm10":("shape",10),"env_svm1":("env",1),"env_svm10":("env",10),
     "rel_svm1":("rel",1),"rel_svm10":("rel",10),"rel_lda":("rel",None),
     "peak2_svm1":("peak2",1)}
GPU={"spectral1s":"spectral1s","spectral2s":"spectral2s",
     "envelope_abs":"envelope_abs","envelope_relative":"envelope_relative",
     "envelope_hybrid":"envelope_hybrid","envelope_multitask":"envelope_hybrid",
     "spectral_multitask":"spectral1s"}


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(".tmp");t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n");t.replace(p)
def score(y,p):
    cm=np.bincount(y*8+p,minlength=64).reshape(8,8)
    tp=np.diag(cm);precision=np.divide(tp,cm.sum(0),out=np.zeros(8),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(8),where=cm.sum(1)>0)
    f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(8),where=cm.sum(0)+cm.sum(1)>0)
    return {"accuracy":float(np.mean(y==p)),"macro_f1":float(f1.mean()),
            "macro_precision":float(precision.mean()),"macro_recall":float(recall.mean()),
            "gesture_accuracy":float(np.mean(y//2==p//2)),
            "angle_accuracy":float(np.mean(y%2==p%2)),"confusion_matrix":cm.tolist(),"n":len(y)}


def splits():return dict(np.load(DATA/"splits.npz"))
def freeze():
    record={"source_hashes":{n:sha(HERE/n) for n in ["common.py","cpu.py","gpu.py","prepare.py"]},
            "protocol_sha256":sha(HERE/"PROTOCOL.md"),"data_hashes":json.loads((DATA/"sha256.json").read_text()),
            "cpu_candidates":CPU,"gpu_candidates":GPU}
    # JSON round trip normalizes tuples before comparison on resumed runs.
    record=json.loads(json.dumps(record))
    path=OUT/"results/RUN_FROZEN.json"
    if path.exists():assert json.loads(path.read_text())["frozen"]==record,"Run source/config changed"
    else:write(path,{"timestamp":stamp(),"frozen":record})

def assert_groups(s,tr,va):
    assert not np.any(tr&va)
    assert set(s["recording"][tr]).isdisjoint(s["recording"][va])
    assert set(s["trial"][tr]).isdisjoint(s["trial"][va])
    assert np.all(np.bincount(s["y"][tr],minlength=8)>0)
    assert not np.any(s["test"]&(tr|va))


def cases(stage,name):
    s=splits();dev=s["development"];fold=s["fold"]
    if stage=="screen":
        for k in range(5):
            tr,va=dev&(fold!=k),dev&(fold==k);assert_groups(s,tr,va)
            yield OUT/f"results/screen/{name}/fold{k}",tr,va,42+k,None
    elif stage=="nested_inner":
        for outer in range(5):
            pool=dev&(fold!=outer);f=s[f"inner{outer}"]
            for k in range(3):
                tr,va=pool&(f!=k),pool&(f==k);assert_groups(s,tr,va)
                yield OUT/f"results/nested_inner/outer{outer}/{name}/fold{k}",tr,va,4242+10*outer+k,None
    elif stage=="nested_outer":
        lock=json.loads((OUT/"results/NESTED_SELECTION_LOCK.json").read_text())
        for k,item in enumerate(lock["folds"]):
            if name not in {item["selected"],"spectral1s"}:continue
            tr,va=dev&(fold!=k),dev&(fold==k);assert_groups(s,tr,va)
            epoch=item["candidates"][name]["epochs"]
            yield OUT/f"results/nested_outer/outer{k}/{name}",tr,va,8042+k,epoch
    elif stage=="final":
        lock=json.loads((OUT/"results/FINAL_SELECTION_LOCK.json").read_text())
        if name not in lock["evaluate"]:return
        for seed in ([42,2026] if name in GPU else [42]):
            yield OUT/f"results/final/{name}/seed{seed}",dev,s["test"],seed,lock["candidates"][name]["epochs"]
    else:raise ValueError(stage)


def summarize(folder):
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob("fold*/summary.json"))]
    assert len(rows) in (3,5),(folder,len(rows))
    return {"accuracy":float(np.mean([r["accuracy"] for r in rows])),
            "macro_f1":float(np.mean([r["macro_f1"] for r in rows])),
            "accuracy_sd":float(np.std([r["accuracy"] for r in rows],ddof=1)),
            "epochs":max(1,int(np.floor(np.median([r.get("best_epoch",1) for r in rows])+.5))),
            "folds":rows}


def select():
    candidates=list(CPU)+list(GPU)
    nested=[]
    for outer in range(5):
        all_results={n:summarize(OUT/f"results/nested_inner/outer{outer}/{n}") for n in candidates}
        selected=sorted(candidates,key=lambda n:(-all_results[n]["macro_f1"],n))[0]
        nested.append({"outer":outer,"selected":selected,"candidates":all_results})
    all_screen={n:summarize(OUT/f"results/screen/{n}") for n in candidates}
    selected=sorted(candidates,key=lambda n:(-all_screen[n]["macro_f1"],n))[0]
    matched=[n for n in candidates if n in ["legacy_svm1","shape_svm1","shape_svm10","spectral1s","spectral_multitask"]]
    selected1s=sorted(matched,key=lambda n:(-all_screen[n]["macro_f1"],n))[0]
    frozen={"timestamp":stamp(),"source_hashes":{p.name:sha(p) for p in HERE.glob("*.py")},
            "protocol_sha256":sha(HERE/"PROTOCOL.md"),"split_sha256":sha(DATA/"splits.npz")}
    for name,value in [("NESTED_SELECTION_LOCK",{"folds":nested}),
                       ("FINAL_SELECTION_LOCK",{"selected":selected,"selected_matched_1s":selected1s,
                         "evaluate":sorted(set([selected,selected1s,"spectral1s","legacy_svm1"])),
                         "candidates":all_screen})]:
        path=OUT/f"results/{name}.json"
        assert not path.exists(),"Selection already frozen"
        write(path,{**frozen,**value})
    print("FROZEN",selected,selected1s,[r["selected"] for r in nested],flush=True)


if __name__=="__main__":select()
