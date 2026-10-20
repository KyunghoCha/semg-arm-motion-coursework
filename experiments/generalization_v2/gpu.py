"""Bounded CUDA models with train-only normalization and saved predictions."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import sys,time,json,random,shutil,subprocess
import numpy as np
import torch
from torch import nn
from common import *
sys.path.insert(0,str(OLD))
import train_gpu as old


class Model(nn.Module):
    def __init__(self,name,mean,scale):
        super().__init__();self.name=name;self.multi="multitask" in name
        self.register_buffer("mean",torch.tensor(mean,dtype=torch.float32))
        self.register_buffer("scale",torch.tensor(scale,dtype=torch.float32))
        if name.startswith("spectral"):
            self.body=nn.Sequential(nn.Conv1d(165,64,1),nn.BatchNorm1d(64),nn.LeakyReLU(.1),
                    nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.LeakyReLU(.1),
                    nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.LeakyReLU(.1))
            size=128
        else:
            channels=mean.shape[1]
            self.body=nn.Sequential(nn.Conv1d(channels,32,5,padding=2),nn.GroupNorm(4,32),nn.GELU(),
                  nn.Conv1d(32,32,5,padding=2),nn.GroupNorm(4,32),nn.GELU(),
                  nn.Conv1d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU())
            size=160
        self.head=nn.Sequential(nn.Linear(size,64),nn.LeakyReLU(.1),nn.Dropout(.25),nn.Linear(64,8))
        if self.multi:
            self.gesture=nn.Linear(size,4);self.angle=nn.Linear(size,2)

    def forward(self,x,aux=False):
        h=self.body((x-self.mean)/self.scale)
        h=torch.cat([h.mean(-1),h.amax(-1)],1) if self.name.startswith("spectral") else h.reshape(len(h),32,5,10).mean(-1).flatten(1)
        logits=self.head(h)
        return (logits,self.gesture(h),self.angle(h)) if aux and self.multi else logits


def resource(start):
    assert time.monotonic()-start<5400,"90-minute GPU budget"
    assert shutil.disk_usage(OUT).free>5*1024**3,"low disk"
    row=subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu,memory.used","--format=csv,noheader,nounits"],text=True)
    temp,used=map(int,row.strip().split(","))
    assert temp<85 and torch.cuda.max_memory_allocated()<8*1024**3
    return {"gpu_temp_c":temp,"device_memory_mib":used,"allocated_mib":torch.cuda.max_memory_allocated()/1024**2}


@torch.no_grad()
def predict(model,x,indices):
    model.eval()
    return np.concatenate([model(x[b]).softmax(1).cpu().numpy() for b in torch.as_tensor(indices,device="cuda").split(128)])


def fit(name,array,x,y,tr,va,folder,seed,start,epochs=None):
    old.set_seed(seed);torch.cuda.reset_peak_memory_stats()
    mean,scale=old.normalization(array,tr);model=Model(name,mean,scale).cuda()
    assert sum(p.numel() for p in model.parameters())<250000
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    weight=None if name=="spectral1s" else torch.as_tensor(len(y[tr])/(8*np.bincount(y[tr],minlength=8)),dtype=torch.float32,device="cuda")
    criterion=nn.CrossEntropyLoss(weight=weight,label_smoothing=.05)
    auxcriterion=nn.CrossEntropyLoss(label_smoothing=.05)
    yg=torch.as_tensor(y,device="cuda");train=torch.as_tensor(np.flatnonzero(tr),device="cuda")
    folder.mkdir(parents=True,exist_ok=True)
    maximum,minimum,patience=(40,8,8) if name=="spectral1s" else (60,12,12)
    best=-1;stale=0;best_epoch=0;t0=time.monotonic()
    with (folder/"epochs.jsonl").open("w") as log:
        for epoch in range(1,(epochs or maximum)+1):
            model.train();total=0;correct=0;count=0
            for b in train[torch.randperm(len(train),device="cuda")].split(128):
                opt.zero_grad(set_to_none=True)
                if model.multi:
                    z,g,a=model(x[b],aux=True)
                    loss=criterion(z,yg[b])+.3*auxcriterion(g,yg[b]//2)+.3*auxcriterion(a,yg[b]%2)
                else:z=model(x[b]);loss=criterion(z,yg[b])
                assert torch.isfinite(loss);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
                total+=loss.item()*len(b);correct+=int((z.argmax(1)==yg[b]).sum());count+=len(b)
            row={"epoch":epoch,"timestamp":stamp(),"train_loss":total/count,"train_accuracy":correct/count}
            if epochs is None:
                prob=predict(model,x,np.flatnonzero(va));m=score(y[va],prob.argmax(1))
                row.update(val_accuracy=m["accuracy"],val_macro_f1=m["macro_f1"])
                if m["macro_f1"]>best+1e-12:
                    best=m["macro_f1"];best_epoch=epoch;stale=0
                    torch.save(model.state_dict(),folder/"best.pt")
                else:stale+=1
            row.update(resource(start));log.write(json.dumps(row)+"\n");log.flush()
            if epochs is None and epoch>=minimum and stale>=patience:break
    torch.save(model.state_dict(),folder/"last.pt")
    if epochs is None:model.load_state_dict(torch.load(folder/"best.pt",weights_only=True))
    else:best_epoch=epochs
    prob=predict(model,x,np.flatnonzero(va))
    np.savez_compressed(folder/"validation.npz",indices=np.flatnonzero(va),true=y[va],probabilities=prob)
    result={**score(y[va],prob.argmax(1)),"best_epoch":best_epoch,"epochs_run":epoch,
            "seconds":time.monotonic()-t0,"seed":seed,"name":name,"timestamp":stamp(),
            "parameters":sum(p.numel() for p in model.parameters()),**resource(start)}
    write(folder/"summary.json",result)
    return result


def self_check():
    old.set_seed(7)
    result={}
    for name,key in GPU.items():
        channels=165 if name.startswith("spectral") else 10 if key=="envelope_hybrid" else 5
        x=torch.randn(24,channels,59 if channels==165 else 50,device="cuda")
        model=Model(name,np.zeros((1,channels,1)),np.ones((1,channels,1))).cuda()
        y=torch.arange(24,device="cuda")%8
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        for _ in range(15):
            opt.zero_grad()
            if model.multi:
                z,g,a=model(x,aux=True)
                loss=nn.functional.cross_entropy(z,y)+.3*nn.functional.cross_entropy(g,y//2)+.3*nn.functional.cross_entropy(a,y%2)
            else:z=model(x);loss=nn.functional.cross_entropy(z,y)
            loss.backward();opt.step()
        model.eval()
        with torch.no_grad():z=model(x)
        assert z.shape==(24,8) and torch.isfinite(z).all()
        state={k:v.clone() for k,v in model.state_dict().items()}
        second=Model(name,np.zeros((1,channels,1)),np.ones((1,channels,1))).cuda();second.load_state_dict(state);second.eval()
        with torch.no_grad():assert torch.equal(z,second(x))
        result[name]={"shape_and_finite":True,"state_replay_exact":True,"toy_accuracy":float((z.argmax(1)==y).float().mean())}
    write(OUT/"reports/gpu_self_check.json",result);print(result,flush=True)


def run(stage):
    assert torch.cuda.is_available(),"User GPU required"
    torch.set_num_threads(2)
    if stage=="self-check":return self_check()
    freeze()
    s=splits();y=s["y"];start=time.monotonic()
    for name,key in GPU.items():
        array=np.load(DATA/f"{key}.npy");x=torch.from_numpy(array).cuda()
        for folder,tr,va,seed,epochs in cases(stage,name):
            if (folder/"summary.json").exists():continue
            r=fit(name,array,x,y,tr,va,folder,seed,start,epochs)
            print(stage,str(folder.relative_to(OUT)),f"acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f}",flush=True)
        del x;torch.cuda.empty_cache()


if __name__=="__main__":run(sys.argv[1])
