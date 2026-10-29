#!/usr/bin/env python3
"""E03-A frozen ImageNet ResNet18 volume embeddings + MIL heads.
Exploratory: uses the existing GAMMA 99-case development folds. Not external validation.
Never writes under results/e01_cv or results/e01_dedup.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path('/home/test/OpticNeuroFusion_Starter')
SPLITS = BASE / 'results/e01_dedup/manifests'
OUTPUT = BASE / 'results/e03_a'
FEATURES = OUTPUT / 'features_imagenet_resnet18_v1'
SEEDS = (42, 43, 44)
MODES = ('oct_mean', 'oct_attn', 'concat', 'gated')


def manifest(fold=1, seed=42):
    p = SPLITS / f'fold{fold}_seed{seed}.csv'
    df = pd.read_csv(p, dtype={'patient_id': str})
    df.patient_id = df.patient_id.str.zfill(4)
    if len(df) != 99 or not df.patient_id.is_unique or '0077' in set(df.patient_id):
        raise ValueError(f'Unsafe manifest: {p}')
    return df


def preflight():
    df = manifest()
    ref = dict(zip(df.patient_id, df.label))
    all_ids = set(ref)
    assert len(all_ids) == 99
    for seed in SEEDS:
        hit = {x: 0 for x in all_ids}
        for fold in range(1, 6):
            part = manifest(fold, seed)
            if set(part.patient_id) != all_ids or part.set_index('patient_id').label.to_dict() != ref:
                raise ValueError('ID/label mismatch across manifests')
            for case_id in part.loc[part.split == 'test', 'patient_id']:
                hit[case_id] += 1
            for split in ('train','val','test'):
                if not (part.split == split).any():
                    raise ValueError(f'Missing split: {split}')
        if set(hit.values()) != {1}:
            raise ValueError('Invalid OOF coverage')
    for r in df.itertuples():
        if not Path(r.fundus_path).is_file():
            raise FileNotFoundError(r.fundus_path)
        octs = sorted(Path(r.oct_path).parent.glob('*_image.jpg'), key=lambda p: int(p.stem.split('_')[0]))
        ix = [int(p.stem.split('_')[0]) for p in octs]
        if len(octs) != 256 or ix != list(range(ix[0], ix[0]+256)):
            raise ValueError(f'Incomplete OCT: {r.patient_id}')
    print('PREFLIGHT PASS: 99 cases, 15 splits, 256 B-scans/case, OOF coverage', flush=True)
    return df


def hash_audit():
    df = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i,r in enumerate(df.itertuples(),1):
        fh = hashlib.sha256(Path(r.fundus_path).read_bytes()).hexdigest()
        oct_dir = Path(r.oct_path).parent
        slices = sorted(oct_dir.glob('*_image.jpg'), key=lambda p: int(p.stem.split('_')[0]))
        full = hashlib.sha256()
        for p in slices:
            full.update(hashlib.sha256(p.read_bytes()).digest())
        rows.append({'case_id':r.patient_id,'fundus_hash':fh,'volume_hash':full.hexdigest()})
        if i % 10 == 0: print('HASHED',i,'/',len(df),flush=True)
    hashes=pd.DataFrame(rows)
    hashes.to_csv(OUTPUT/'full_volume_hash_audit.csv',index=False)
    bad=hashes[hashes.duplicated('fundus_hash',keep=False) | hashes.duplicated('volume_hash',keep=False)]
    if len(bad):
        print('DUPLICATE PAIRS FOUND:',bad.to_string(index=False),flush=True)
        raise RuntimeError('Resolve cross-case identical fundus/complete OCT volumes before training')
    print('FULL-VOLUME HASH AUDIT PASS: 99 unique fundus and 99 unique volumes',flush=True)


def weights_and_transforms():
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    except Exception as exc:
        raise RuntimeError('Check torchvision compatibility with installed torch') from exc
    weights = ResNet18_Weights.IMAGENET1K_V1
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError('Could not retrieve pretrained ImageNet weights: no silent random init') from exc
    model.fc = nn.Identity()
    transform = Compose([
        Resize((224,224)), ToTensor(),
        Normalize(mean=(.485,.456,.406), std=(.229,.224,.225))
    ])
    return model, transform


def feature_extraction(batch_size=32, limit=0):
    df = preflight()
    model, transform = weights_and_transforms()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
    FEATURES.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(df.itertuples(), 1):
        if limit and i > limit: break
        dst = FEATURES / f'{r.patient_id}.pt'
        if dst.exists():
            stored = torch.load(dst, map_location='cpu', weights_only=True)
            if stored['oct'].shape == (256,512) and stored['fundus'].shape == (512,):
                print(f'SKIP {r.patient_id}', flush=True); continue
        files = sorted(Path(r.oct_path).parent.glob('*_image.jpg'), key=lambda p: int(p.stem.split('_')[0]))
        if len(files) != 256: raise ValueError('Insufficient OCT')
        all_paths = [Path(r.fundus_path)] + files
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(all_paths), batch_size):
                tensors = []
                for path in all_paths[start:start+batch_size]:
                    with Image.open(path) as im:
                        tensors.append(transform(im.convert('RGB')))
                chunks.append(model(torch.stack(tensors).to(device)).cpu())
        emb = torch.cat(chunks)
        if emb.shape != (257,512): raise ValueError('Feature shape mismatch')
        tmp = dst.with_suffix('.tmp')
        torch.save({'case_id':r.patient_id, 'label':int(r.label),
                    'fundus':emb[0].contiguous(), 'oct':emb[1:].contiguous()}, tmp)
        tmp.replace(dst)
        print(f'CACHED {i}/{len(df)} {r.patient_id}', flush=True)
    print('FEATURE EXTRACTION DONE', flush=True)


def get_positions(n):
    if n == 1: return np.asarray([128])
    if n < 1 or n > 256: raise ValueError('Invalid slice count')
    ids = np.rint(np.linspace(0,255,n)).astype(int)
    assert len(np.unique(ids)) == n
    return ids


class Head(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode=mode
        self.oct_proj=nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(.25))
        self.fundus_proj=nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(.25))
        self.att=nn.Sequential(nn.Linear(513,96), nn.Tanh(), nn.Linear(96,1))
        self.gate=nn.Linear(256,128)
        self.combine=nn.Sequential(nn.Linear(256,128), nn.ReLU(),nn.Dropout(.25))
        self.classifier=nn.Linear(128,3)
    def forward(self, fundus, octs):
        if self.mode == 'oct_mean':
            ov=octs.mean(dim=1)
        else:
            B,S,D = octs.shape
            positions=torch.linspace(-1,1,S,device=octs.device).view(1,S,1).expand(B,-1,-1)
            weight=torch.softmax(self.att(torch.cat([octs,positions],dim=-1)).squeeze(-1),dim=1)
            ov=(weight.unsqueeze(-1)*octs).sum(dim=1)
        o=self.oct_proj(ov)
        if self.mode in ('oct_mean','oct_attn'):
            z=o
        else:
            f=self.fundus_proj(fundus)
            together=torch.cat([o,f],dim=-1)
            if self.mode=='concat': z=self.combine(together)
            elif self.mode=='gated':
                gate=torch.sigmoid(self.gate(together))
                z=gate*f+(1-gate)*o
            else: raise ValueError(self.mode)
        return self.classifier(z)


def load_features():
    df=manifest()
    arr={}
    for r in df.itertuples():
        p=FEATURES / f'{r.patient_id}.pt'
        if not p.exists(): raise FileNotFoundError(f'Run --stage cache first: {p}')
        v=torch.load(p,map_location='cpu',weights_only=True)
        if v['label']!=int(r.label) or tuple(v['oct'].shape)!=(256,512):
            raise RuntimeError(f'Invalid feature cache: {p}')
        arr[r.patient_id]=v
    return arr


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_one(features, fold, seed, n_slices, mode, epochs, batch):
    exp=f'fold{fold}_seed{seed}_n{n_slices}_{mode}'
    dest=OUTPUT / 'runs' / exp
    complete=dest/'complete.json'
    if complete.exists():
        print('SKIP',exp,flush=True);return
    part=manifest(fold,seed)
    positions=get_positions(n_slices)
    def collect(split):
        rows=part[part.split==split]
        cases=rows.patient_id.tolist()
        fundus=torch.stack([features[c]['fundus'] for c in cases])
        octs=torch.stack([features[c]['oct'][positions] for c in cases])
        y=torch.tensor(rows.label.to_numpy(dtype=int),dtype=torch.long)
        return cases,fundus,octs,y
    train=collect('train'); val=collect('val'); test=collect('test')
    assert not (set(train[0])&set(val[0]) or set(train[0])&set(test[0]) or set(val[0])&set(test[0]))
    set_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=Head(mode).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
    # Inner-val loss only; no outer-test selection.
    count=np.bincount(train[3].numpy(),minlength=3)
    weights=torch.tensor(len(train[3])/(3*np.maximum(count,1)),dtype=torch.float32,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    dl=DataLoader(TensorDataset(train[1],train[2],train[3]),batch_size=batch,shuffle=True)
    best_loss=float('inf');best_state=None;best_epoch=0;wait=0
    vf,vo,vy=[x.to(device) for x in val[1:]]
    for epoch in range(1,epochs+1):
        net.train()
        for f,o,y in dl:
            optim.zero_grad(set_to_none=True)
            loss=loss_fn(net(f.to(device),o.to(device)),y.to(device))
            loss.backward();optim.step()
        net.eval()
        with torch.inference_mode():
            vl=float(loss_fn(net(vf,vo),vy).item())
        if vl < best_loss - 1e-5:
            best_loss=vl;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch;wait=0
        else:
            wait+=1
        if wait>=12:break
    net.load_state_dict(best_state);net.eval()
    tf,to,ty=[x.to(device) for x in test[1:]]
    with torch.inference_mode():
        p=torch.softmax(net(tf,to),dim=1).cpu().numpy()
    y=test[3].numpy();pred=p.argmax(1)
    try: auroc=float(roc_auc_score(y,p,multi_class='ovr',average='macro',labels=[0,1,2]))
    except ValueError: auroc=None
    rec={'experiment':exp,'fold':fold,'seed':seed,'n_slices':n_slices,'mode':mode,
         'n':len(y),'n_train':len(train[0]),'n_val':len(val[0]),
         'best_epoch':best_epoch,'val_loss':best_loss,
         'macro_auc_ovr':auroc,
         'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
         'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
         'accuracy':float(accuracy_score(y,pred)),
         'trainable_parameters':sum(t.numel() for t in net.parameters() if t.requires_grad)}
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'case_id':cid,'y_true':int(y[i]),'y_pred':int(pred[i]),
                   'p0':float(p[i,0]),'p1':float(p[i,1]),'p2':float(p[i,2])}
                  for i,cid in enumerate(test[0])]).to_csv(dest/'predictions.csv',index=False)
    torch.save(best_state,dest/'best.pt')
    (dest/'metrics.json').write_text(json.dumps(rec,indent=2),encoding='utf8')
    complete.write_text(json.dumps({'complete':True,'experiment':exp}),encoding='utf8')
    print('COMPLETE',exp,'AUROC',auroc,'val_loss',round(best_loss,4),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('preflight','hash','cache','train','summary'),required=True)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--epochs',type=int,default=45)
    ap.add_argument('--limit-cases',type=int,default=0)
    ap.add_argument('--limit-runs',type=int,default=0)
    ap.add_argument('--slices',nargs='+',type=int,default=[1,8,32,256])
    ap.add_argument('--modes',nargs='+',default=list(MODES))
    args=ap.parse_args()
    if args.stage=='preflight': preflight();return
    if args.stage=='hash': hash_audit();return
    if args.stage=='cache': feature_extraction(batch_size=args.batch,limit=args.limit_cases);return
    if args.stage=='train':
        preflight()
        features=load_features()
        runs=[(fold,seed,n,mode) for fold in range(1,6) for seed in SEEDS for n in args.slices for mode in args.modes]
        print('PLANNED',len(runs),'runs',flush=True)
        for ix,(fold,seed,n,mode) in enumerate(runs):
            if args.limit_runs and ix>=args.limit_runs:break
            run_one(features,fold,seed,n,mode,args.epochs,args.batch)
        return
    rows=[]
    for p in sorted((OUTPUT/'runs').glob('*/metrics.json')):
        rows.append(json.loads(p.read_text()))
    if not rows:print('NO COMPLETED RUNS');return
    df=pd.DataFrame(rows)
    print('COMPLETED:',len(df))
    print(df.groupby(['n_slices','mode'])[['macro_auc_ovr','macro_f1','balanced_accuracy']].agg(['count','mean','std']).round(4))
    df.to_csv(OUTPUT/'e03_a_metrics.csv',index=False)


if __name__=='__main__':main()
