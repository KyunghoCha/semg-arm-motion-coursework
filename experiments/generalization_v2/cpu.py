"""CPU feature controls. New sources/settings are isolated from old runs."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import sys,time,json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from common import *


def run(stage):
    freeze()
    s=splits();y=s["y"]
    for name,(feature,c) in CPU.items():
        x=np.load(DATA/f"cpu_{feature}.npy")
        for folder,tr,va,seed,epochs in cases(stage,name):
            if (folder/"summary.json").exists():continue
            model=make_pipeline(StandardScaler(),SVC(C=c,gamma="scale",cache_size=256) if c else
                                LinearDiscriminantAnalysis(solver="lsqr",shrinkage=.5))
            t=time.monotonic();model.fit(x[tr],y[tr])
            fit_seconds=time.monotonic()-t;p=model.predict(x[va])
            result={**score(y[va],p),"fit_seconds":fit_seconds,"timestamp":stamp(),"name":name,
                    "seed":seed,"train_indices_sha256":hashlib.sha256(np.flatnonzero(tr).tobytes()).hexdigest()}
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/"validation.npz",indices=np.flatnonzero(va),true=y[va],pred=p)
            if stage=="final":
                import joblib
                joblib.dump(model,folder/"model.joblib")
                t=time.monotonic()
                for _ in range(100):model.predict(x[va][:1])
                result["single_inference_ms"]=(time.monotonic()-t)*10
            write(folder/"summary.json",result)
            print(stage,str(folder.relative_to(OUT)),f"acc={result['accuracy']:.4f} f1={result['macro_f1']:.4f}",flush=True)


if __name__=="__main__":run(sys.argv[1])
