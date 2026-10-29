#!/usr/bin/env python3
"""E04 pre-specified, label-blind inference then one-time evaluation.
Train set is NEVER used to tune E04 models; NO train operation exists.
External FairVision Glaucoma Test 3000, 15 GAMMA checkpoints per model.
Run: --stage lock | prepare | infer | evaluate | status | selftest
"""
import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, brier_score_loss,
                             balanced_accuracy_score)

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / 'data/external/fairvision/Glaucoma/Dataset/dataset.zip'
CSV = ROOT / 'data/external/fairvision/Glaucoma/ReadMe/data_summary_glaucoma.csv'
PAIRS = ROOT / 'results/e04/fairvision_pairs_audit.csv'
OUT = ROOT / 'results/e04/formal_v1'
MODEL_DIR = OUT / 'individual_predictions'
CONFIGS = ('octlast_concat', 'octlast_cross', 'oct_attn_frozen')
SEEDS = (42,43,44)
FOLDS = (1,2,3,4,5)
N = 3000
B = 5000
SLICES = np.rint(np.linspace(0,199,8)).astype(int)
PREP = OUT / 'test_rgb224_uint8.npy'
INDEX = OUT / 'test_index.csv'
PROTOCOL = OUT / 'protocol.json'
MEAN = np.array([.485,.456,.406], dtype=np.float32)
STD = np.array([.229,.224,.225], dtype=np.float32)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda:f.read(16*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def atomic_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
    os.replace(tmp, p)


def decode_label(value):
    if isinstance(value, bytes): value=value.decode('utf8')
    v=str(value).strip().lower()
    if v in ('no','0','0.0'): return 0
    if v in ('yes','1','1.0'): return 1
    raise ValueError(f'Unknown glaucoma label in NPZ: {value!r}')

def metadata():
    d = pd.read_csv(CSV, dtype={'filename':str})
    labels = d.glaucoma.astype(str).str.strip().str.lower().map({'no':0,'yes':1})
    if labels.isna().any():
        raise ValueError('Unknown glaucoma label')
    d['label'] = labels.astype(int)
    d['case_id'] = d.filename.str.extract(r'(\d{5})\.npz$', expand=False)
    d['split'] = d['use'].astype(str).str.lower().str.strip()
    if len(d)!=10000 or not d.case_id.is_unique or d.case_id.isna().any():
        raise ValueError('Metadata ID/size mismatch')
    if d.split.value_counts().to_dict() != {'training':6000,'validation':1000,'test':3000}:
        raise ValueError('Official splits unexpected')
    return d


def get_test_index():
    d = metadata()
    p = pd.read_csv(PAIRS, dtype={'case_id':str})
    if len(p)!=10000 or not p.case_id.is_unique:
        raise ValueError('Pairs file invalid')
    t = d.loc[d.split=='test', ['case_id','label']].merge(
        p.loc[p.split.str.lower()=='test',['case_id','npz_path']],
        on='case_id', how='left',validate='one_to_one')
    t = t.sort_values('case_id').reset_index(drop=True)
    if len(t)!=N or t.npz_path.isna().any() or t.case_id.duplicated().any():
        raise ValueError('Test index invalid')
    if int((t.label==0).sum())!=1534 or int((t.label==1).sum())!=1466:
        raise ValueError('Test label counts mismatch')
    if not t.npz_path.str.startswith('Test/').all():
        raise ValueError('Non-test NPZ supplied')
    return t


def checkpoints():
    result={}
    for conf in CONFIGS:
        for fold in FOLDS:
            for seed in SEEDS:
                if conf == 'oct_attn_frozen':
                    d = ROOT / f'results/e03_a/runs/fold{fold}_seed{seed}_n8_oct_attn'
                else:
                    d = ROOT / f'results/e03_c/runs/fold{fold}_seed{seed}_{conf}'
                if not (d/'complete.json').is_file() or not (d/'best.pt').is_file():
                    raise FileNotFoundError(f'Incomplete checkpoint {d}')
                result[f'{conf}_fold{fold}_seed{seed}'] = d/'best.pt'
    if len(result)!=45:
        raise ValueError('Expected 45 checkpoints')
    return result


def script_hash():
    return digest(Path(__file__).resolve())


def lock():
    if PROTOCOL.exists():
        raise RuntimeError('Protocol already locked; refuse overwrite')
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Non-empty output dir without protocol; inspect it')
    t = get_test_index()
    ck = checkpoints()
    with zipfile.ZipFile(ZIP) as z:
        names=set(z.namelist())
        if not set(t.npz_path).issubset(names):
            raise RuntimeError('Missing Test NPZ files in ZIP')
    import torch, torchvision
    result={
      'version':'e04_fairvision_formal_v1',
      'created_at':datetime.now().astimezone().isoformat(),
      'data':'Harvard-FairVision Glaucoma official Test only',
      'test_cases':N, 'test_normal':1534,'test_glaucoma':1466,
      'dataset_zip_sha256':digest(ZIP),
      'metadata_sha256':digest(CSV),
      'pairs_sha256':digest(PAIRS),
      'source_sha256':script_hash(),
      'checkpoints_sha256':{k:digest(v) for k,v in ck.items()},
      'positions':SLICES.tolist(),
      'oct_axis':0,
      'axis_rationale':'first dimension is B-scan count: 200 glaucoma, 128 AMD/DR in official data card; anatomical orientation not fully verified',
      'slo':'NPZ slo_fundus 200x200 grayscale converted to 3-channel RGB',
      'oct':'NPZ oct_bscans 200x200x200, axis0, no flip/rotate; 8 uniformly selected',
      'resize':'PIL bilinear 224x224, no crop, no contrast change',
      'normalization':{'mean':MEAN.tolist(),'std':STD.tolist()},
      'models':list(CONFIGS),
      'ensemble':'arithmetic mean of 15 per-model 3-class softmax vectors',
      'binary_score':'p1+p2 = 1-p0',
      'main_metric':'binary AUROC, C4 vs C1',
      'supporting_metrics':['AUPRC','sensitivity@0.5','specificity@0.5','balanced_accuracy@0.5','Brier'],
      'exploratory_subgroups':['race','gender','age (<60 vs >=60)'],
      'inference_no_training':True,
      'bootstrap':{'replicates':B,'seed':20261027,'unit':'FairVision case','scheme':'stratified paired by label'},
      'versions':{'python':platform.python_version(),'torch':torch.__version__,
                  'torchvision':torchvision.__version__,'numpy':np.__version__,
                  'pandas':pd.__version__},
      'warnings':['SLO vs color fundus domain shift','OCT geometry not confirmed anatomical equivalence',
                  'GAMMA case IDs not proven unique patients',
                  'OCT-only baseline uses frozen, unlike OCT fine-tuned C1/C4']
    }
    OUT.mkdir(parents=True,exist_ok=True)
    t[['case_id','npz_path']].to_csv(INDEX,index=False)
    atomic_json(PROTOCOL,result)
    print('LOCK PASS: 3,000 Test cases, 45 checkpoints, ZIP and source hashed',flush=True)
    print('LOCK:',PROTOCOL,flush=True)


def check_locked(check_hashes=True):
    if not PROTOCOL.exists():
        raise RuntimeError('Run --stage lock first')
    p=json.loads(PROTOCOL.read_text())
    if script_hash()!=p['source_sha256']:
        raise RuntimeError('Code changed since protocol lock')
    if digest(CSV)!=p['metadata_sha256'] or digest(PAIRS)!=p['pairs_sha256']:
        raise RuntimeError('Dataset metadata changed since lock')
    current=get_test_index()[['case_id','npz_path']]
    saved=pd.read_csv(INDEX,dtype={'case_id':str})
    if not saved.equals(current):
        raise RuntimeError('Test index changed')
    ck=checkpoints()
    if set(ck)!=set(p['checkpoints_sha256']):
        raise RuntimeError('Model roster differs from protocol')
    if check_hashes:
        for k,path in ck.items():
            if digest(path)!=p['checkpoints_sha256'][k]:
                raise RuntimeError('Checkpoint modified: '+k)
    return p,saved,ck


def image_rgb224(a):
    if a.dtype!=np.uint8 or a.shape!=(200,200):
        raise ValueError('Unexpected image dtype/shape')
    pic=Image.fromarray(a).convert('RGB')
    pic=pic.resize((224,224),resample=Image.Resampling.BILINEAR)
    return np.asarray(pic,dtype=np.uint8).transpose(2,0,1)


def elapsed_string(s):
    return time.strftime('%H:%M:%S',time.gmtime(max(0,int(s))))


def prepare():
    protocol,t,ck=check_locked()
    if PREP.exists():
        a=np.load(PREP,mmap_mode='r')
        if a.shape!=(N,9,3,224,224) or a.dtype!=np.uint8:
            raise RuntimeError('Existing prep has wrong shape/dtype')
        print('PREPARED ALREADY',a.shape,flush=True)
        return
    tmp=OUT/'test_rgb224_uint8.partial.npy'
    progress=OUT/'prepare_progress.json'
    if tmp.exists():
        a=np.lib.format.open_memmap(tmp, mode='r+')
        if a.shape!=(N,9,3,224,224):
            raise RuntimeError('Partial data shape wrong')
        if not progress.exists():
            raise RuntimeError('Partial file with no progress; inspect manually')
        start=int(json.loads(progress.read_text())['done'])
    else:
        a=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(N,9,3,224,224))
        start=0
    t0=time.perf_counter()
    labels=metadata().set_index('case_id')['label']
    with zipfile.ZipFile(ZIP) as z:
        for i in range(start,N):
            row=t.iloc[i]
            with np.load(io.BytesIO(z.read(row.npz_path)), allow_pickle=False) as data:
                octs=np.asarray(data['oct_bscans'])
                slo=np.asarray(data['slo_fundus'])
                y=decode_label(np.asarray(data['glaucoma']).item())
                if octs.shape!=(200,200,200) or octs.dtype!=np.uint8:
                    raise ValueError(f'Invalid OCT {row.case_id}: {octs.shape} {octs.dtype}')
                if slo.shape!=(200,200) or slo.dtype!=np.uint8:
                    raise ValueError(f'Invalid SLO {row.case_id}')
                if y!=int(labels[row.case_id]):
                    raise ValueError(f'CSV/NPZ label mismatch {row.case_id}')
                a[i,0]=image_rgb224(slo)
                for j,pos in enumerate(SLICES,1):
                    a[i,j]=image_rgb224(octs[int(pos)])
            if (i+1)%20==0 or (i+1)==N:
                a.flush()
                atomic_json(progress,{'done':i+1,'total':N})
                dur=time.perf_counter()-t0
                eta=dur/(i+1-start)* (N-i-1)
                print(f'PREPARED {i+1}/{N} | ETA {elapsed_string(eta)}',flush=True)
    del a
    os.replace(tmp,PREP)
    atomic_json(OUT/'prepare_complete.json',{'cases':N,'shape':[N,9,3,224,224]})
    if progress.exists(): progress.unlink()
    print('FULL TEST INPUT PREPARATION COMPLETE',flush=True)


def infer(batch):
    p,t,ck=check_locked()
    if not (OUT/'prepare_complete.json').exists():
        raise RuntimeError('Run --stage prepare first')
    a=np.load(PREP,mmap_mode='r')
    if a.shape!=(N,9,3,224,224) or a.dtype!=np.uint8:
        raise RuntimeError('Input cache mismatch')
    import torch
    from e03c_train import Model
    import e03a_volume_mil as old
    torch.set_num_threads(min(8,os.cpu_count() or 4))
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('CUDA required for full formal run')
    mt=torch.tensor(MEAN,device=device).view(1,1,3,1,1)
    st=torch.tensor(STD,device=device).view(1,1,3,1,1)
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    tasks=list(ck.items())
    def valid_saved(path):
        if not path.exists(): return False
        x=np.load(path, mmap_mode='r')
        if x.shape!=(N,3) or not np.isfinite(x).all() or not np.allclose(x.sum(axis=1),1,atol=2e-5):
            raise RuntimeError('Invalid completed prediction: '+str(path))
        return True
    t_begin=time.perf_counter()
    for task_idx,(key,ckpath) in enumerate(tasks):
        out=MODEL_DIR/f'{key}.npy'
        if valid_saved(out):
            print('SKIP COMPLETED',key,flush=True)
            continue
        conf=key.rsplit('_fold',1)[0]
        if conf=='oct_attn_frozen':
            net,tf=old.weights_and_transforms()
            net=net.eval().to(device)
            head=old.Head('oct_attn').eval().to(device)
            state=torch.load(ckpath,map_location='cpu',weights_only=True)
            head.load_state_dict(state,strict=True)
        else:
            tune, mode=conf.split('_',1)
            model=Model(tune,mode).eval().to(device)
            state=torch.load(ckpath,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
        preds=np.empty((N,3),dtype=np.float32)
        t0=time.perf_counter()
        with torch.inference_mode():
            for i in range(0,N,batch):
                inputs=np.array(a[i:i+batch],copy=True)
                x=torch.from_numpy(inputs).to(device=device,dtype=torch.float32).div_(255.0)
                x.sub_(mt).div_(st)
                if conf=='oct_attn_frozen':
                    Bn=x.shape[0]
                    f=torch.zeros((Bn,512),device=device)
                    feats=[]
                    for j in range(0,8,4):
                        v=net(x[:,1+j:1+j+4].reshape(-1,3,224,224))
                        feats.append(v.reshape(Bn,-1,512))
                    logits=head(f,torch.cat(feats,dim=1))
                else:
                    logits=model(x[:,0],x[:,1:])
                yhat=torch.softmax(logits,dim=1).cpu().numpy()
                if yhat.shape!=(len(inputs),3) or not np.isfinite(yhat).all():
                    raise RuntimeError('Nonfinite/wrong prediction '+key)
                preds[i:i+len(inputs)]=yhat
                if i%200<batch or i+batch>=N:
                    run_dur=time.perf_counter()-t0
                    eta=run_dur/(i+len(inputs))*(N-i-len(inputs))
                    print(f'INFER {task_idx+1}/{len(tasks)} {key}: {i+len(inputs)}/{N} ETA {elapsed_string(eta)}',flush=True)
        if not np.allclose(preds.sum(axis=1),1,atol=2e-5):
            raise RuntimeError('Not normalized')
        tmp=out.with_suffix('.partial.npy')
        np.save(tmp,preds)
        os.replace(tmp,out)
        del preds
        if conf=='oct_attn_frozen': del net,head
        else: del model
        if device.type=='cuda': torch.cuda.empty_cache()
        remaining=len(tasks)-task_idx-1
        avg=(time.perf_counter()-t_begin)/max(1,task_idx+1)
        print(f'COMPLETE {key} | total checkpoint ETA {elapsed_string(avg*remaining)}',flush=True)
    # No label reading here: all scores must be finalized before evaluation.
    final=pd.DataFrame({'case_id':t.case_id})
    for conf in CONFIGS:
        array=[]
        for fold in FOLDS:
            for seed in SEEDS:
                key=f'{conf}_fold{fold}_seed{seed}'
                path=MODEL_DIR/f'{key}.npy'
                if not valid_saved(path):
                    raise RuntimeError('Missing predictions '+key)
                array.append(np.load(path))
        avg=np.mean(np.stack(array,axis=0),axis=0)
        final[f'{conf}_p0']=avg[:,0]
        final[f'{conf}_p1']=avg[:,1]
        final[f'{conf}_p2']=avg[:,2]
        final[f'{conf}_glaucoma_score']=avg[:,1]+avg[:,2]
    tmp=OUT/'blinded_ensemble.partial.csv'
    final.to_csv(tmp,index=False,float_format='%.10g')
    os.replace(tmp,OUT/'blinded_ensemble.csv')
    atomic_json(OUT/'inference_complete.json',{'cases':N,'checkpoint_predictions':len(tasks),'date':datetime.now().isoformat()})
    print('BLINDED PREDICTIONS COMPLETE: 3,000/3,000 x 3 ensembles',flush=True)


def evaluate():
    p,t,ck=check_locked()
    result_dir=OUT/'final'
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError('Final evaluation already performed; refuse rerun/overwrite')
    if not (OUT/'inference_complete.json').exists():
        raise RuntimeError('All test predictions must be complete before evaluation')
    pred=pd.read_csv(OUT/'blinded_ensemble.csv',dtype={'case_id':str})
    if len(pred)!=N or pred.case_id.duplicated().any() or not pred.case_id.equals(t.case_id):
        raise ValueError('Prediction case IDs not in fixed order')
    original=metadata()
    labels=original.set_index('case_id').loc[t.case_id,'label'].to_numpy(dtype=int)
    s={key:pred[f'{key}_glaucoma_score'].to_numpy(float) for key in CONFIGS}
    if any(not np.isfinite(v).all() or np.min(v)<0 or np.max(v)>1 for v in s.values()):
        raise ValueError('Invalid scores')
    from sklearn.metrics import roc_auc_score, average_precision_score
    results=[]
    for key,score in s.items():
        yy=(score>=.5).astype(int)
        tn,fp,fn,tp=confusion_matrix(labels,yy,labels=[0,1]).ravel()
        results.append({'model':key,'n':len(labels),'auc':float(roc_auc_score(labels,score)),
            'auprc':float(average_precision_score(labels,score)),
            'brier':float(brier_score_loss(labels,score)),
            'sensitivity_0p5':tp/(tp+fn),'specificity_0p5':tn/(tn+fp),
            'balanced_accuracy_0p5':balanced_accuracy_score(labels,yy),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
    rng=np.random.default_rng(20261027)
    strata=[np.flatnonzero(labels==0),np.flatnonzero(labels==1)]
    auc_boot={key:np.empty(B) for key in CONFIGS}
    ap_boot={key:np.empty(B) for key in CONFIGS}
    for j in range(B):
        ix=np.concatenate([rng.choice(x,len(x),replace=True) for x in strata])
        for key in CONFIGS:
            auc_boot[key][j]=roc_auc_score(labels[ix],s[key][ix])
            ap_boot[key][j]=average_precision_score(labels[ix],s[key][ix])
        if (j+1)%1000==0: print('BOOTSTRAP',j+1,'/',B,flush=True)
    for row in results:
        key=row['model']
        row['auc_ci_low'],row['auc_ci_high']=np.percentile(auc_boot[key],[2.5,97.5])
        row['auprc_ci_low'],row['auprc_ci_high']=np.percentile(ap_boot[key],[2.5,97.5])
    pairs=[('octlast_cross','octlast_concat'),
           ('octlast_concat','oct_attn_frozen'),
           ('octlast_cross','oct_attn_frozen')]
    comparisons=[]
    for a,b in pairs:
        delta=roc_auc_score(labels,s[a])-roc_auc_score(labels,s[b])
        lo,hi=np.percentile(auc_boot[a]-auc_boot[b],[2.5,97.5])
        comparisons.append({'comparison':f'{a}_minus_{b}',
                            'delta_auc':delta,'ci_low':lo,'ci_high':hi})
    result_dir.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(results).to_csv(result_dir/'model_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(result_dir/'paired_auc_ci.csv',index=False)
    # Prespecified descriptive subgroups: no model choice or threshold tuning.
    dem=original.set_index('case_id').loc[t.case_id].reset_index(drop=True)
    groups={}
    for col in ('race','gender'):
        if col in dem.columns:
            groups[col]=dem[col].astype(str).to_numpy()
    if 'age' in dem.columns:
        age=pd.to_numeric(dem['age'],errors='coerce')
        groups['age_band']=np.where(age.isna(),'missing',np.where(age<60,'lt60','ge60'))
    subgroup=[]
    for attr,values in groups.items():
        for val in sorted(set(values)):
            ix=np.flatnonzero(values==val)
            if len(ix)<50 or len(np.unique(labels[ix]))<2 or min(np.bincount(labels[ix],minlength=2))<10:
                continue
            for key,score in s.items():
                subgroup.append({'attribute':attr,'group':str(val),'model':key,
                    'n':len(ix),'positives':int(labels[ix].sum()),
                    'auc':float(roc_auc_score(labels[ix],score[ix])),
                    'auprc':float(average_precision_score(labels[ix],score[ix]))})
    pd.DataFrame(subgroup).to_csv(result_dir/'exploratory_subgroups.csv',index=False)
    p_out=pred.copy()
    p_out.insert(1,'y_true',labels)
    p_out.to_csv(result_dir/'test_predictions_and_labels.csv',index=False)
    atomic_json(result_dir/'report.json',{'protocol_sha256':digest(PROTOCOL),
        'predictions_sha256':digest(OUT/'blinded_ensemble.csv'),
        'description':'zero-shot 3-class GAMMA to binary FairVision, no external training',
        'results':results,'contrasts':comparisons,
        'limits':['case IDs are not externally verified as unique person identities',
                  'labels and acquisition modalities differ across cohorts',
                  'OCT-only uses frozen compared with fine-tuned C1/C4']})
    print('\n=== EXTERNAL TEST 3000 FULL RESULTS ===',flush=True)
    print(pd.DataFrame(results).round(4).to_string(index=False),flush=True)
    print('\n=== PAIRED AUROC DIFFERENCES ===',flush=True)
    print(pd.DataFrame(comparisons).round(4).to_string(index=False),flush=True)
    print('FORMAL E04 EXTERNAL EVALUATION COMPLETE',flush=True)


def status():
    print('LOCK',PROTOCOL.exists(),'PREPARED',PREP.exists(),flush=True)
    print('PREDICTION CHECKPOINTS',len(list(MODEL_DIR.glob('*.npy'))),'/45',flush=True)
    print('BLINDED ENSEMBLES',(OUT/'blinded_ensemble.csv').exists(),flush=True)
    print('EVALUATED',(OUT/'final/report.json').exists(),flush=True)


def selftest():
    x=np.arange(200*200,dtype=np.uint8).reshape(200,200)
    a=image_rgb224(x)
    assert a.shape==(3,224,224) and a.dtype==np.uint8
    assert len(SLICES)==8 and len(set(SLICES))==8
    assert int(SLICES[0])==0 and int(SLICES[-1])==199
    r=np.array([[.4,.2,.4],[.1,.5,.4]])
    assert np.allclose(r[:,1]+r[:,2],1-r[:,0])
    assert [decode_label(v) for v in ('no','yes','0','1',0,1)]==[0,1,0,1,0,1]
    print('SELFTEST PASS: PIL preprocessing, OCT positions, probability collapse',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',required=True,choices=['selftest','lock','prepare','infer','evaluate','status'])
    ap.add_argument('--batch',type=int,default=4)
    arg=ap.parse_args()
    if arg.batch<1 or arg.batch>16: raise ValueError('--batch must be 1..16')
    {'selftest':selftest,'lock':lock,'prepare':prepare,
     'infer':lambda:infer(arg.batch),'evaluate':evaluate,'status':status}[arg.stage]()


if __name__=='__main__':
    main()
