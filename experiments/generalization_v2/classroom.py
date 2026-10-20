"""Protocol C: classroom assumptions, never treated as A improvement."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import time,importlib.util
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from common import *


def main():
    s=splits();y=s["y"];features=np.load(DATA/"lecture25.npy")
    frozen={"timestamp":stamp(),"code_sha256":sha(__file__),"split_sha256":sha(DATA/"splits.npz"),
      "protocol":"C classroom random trial; fixed 1.5–3.5s; 19 windows x25 features",
      "models":{"SVM":"RBF C1 gamma scale","RF":"100 trees seed42, n_jobs2"},
      "aggregation":{"window_vote":"train on 200ms windows; trial vote of 19 windows",
                     "trial_mean":"mean of 19 feature vectors before fitting; an explicit input-assembly sensitivity analysis"},
      "trial_mean_is_paper_method":False,
      "xgboost_available":importlib.util.find_spec("xgboost") is not None,
      "xgboost_status":"omitted: not installed on this PC; no installation authorization assumed"}
    target=OUT/"results/classroom";target.mkdir(parents=True,exist_ok=True)
    lock=target/"CONFIG_FROZEN.json"
    assert not lock.exists(),"classroom already started"
    write(lock,frozen)
    summary=[]
    for assembly in ["window_vote","trial_mean"]:
        for model_name in ["SVM","RF"]:
            for split in [*range(5),"holdout"]:
                if split=="holdout":tr,va=s["c_train"],s["c_test"]
                else:tr=np.flatnonzero(s["c_fold"]!=split);va=np.flatnonzero(s["c_fold"]==split)
                assert set(s["trial"][tr]).isdisjoint(s["trial"][va])
                model=make_pipeline(StandardScaler(),SVC(C=1,gamma="scale",cache_size=256) if model_name=="SVM"
                                    else RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=2))
                if assembly=="window_vote":
                    xa=features[tr].reshape(-1,25);ya=np.repeat(y[tr],19);xb=features[va].reshape(-1,25)
                else:xa=features[tr].mean(1);ya=y[tr];xb=features[va].mean(1)
                t=time.monotonic();model.fit(xa,ya);fit_seconds=time.monotonic()-t
                pred=model.predict(xb)
                if assembly=="window_vote":
                    tp=np.array([np.bincount(p,minlength=8).argmax() for p in pred.reshape(-1,19)])
                    wm=score(np.repeat(y[va],19),pred)
                else:tp=pred;wm=None
                row={"assembly":assembly,"model":model_name,"split":split,"trial":score(y[va],tp),
                     "window":wm,"fit_seconds":fit_seconds}
                t=time.monotonic()
                for _ in range(100):model.predict(xb[:1])
                row["single_input_inference_ms"]=(time.monotonic()-t)*10
                path=target/f"{assembly}/{model_name}/{split}";path.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path/"predictions.npz",trial_indices=va,true=y[va],trial_pred=tp,
                                    input_predictions=pred)
                write(path/"summary.json",row);summary.append(row)
                print(assembly,model_name,split,"trial",row["trial"]["accuracy"],"window",None if wm is None else wm["accuracy"],flush=True)
    write(target/"all_results.json",summary)


if __name__=="__main__":main()
