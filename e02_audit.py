#!/usr/bin/env python3
"""Read-only E01 ID/path/hash audit and pooled OOF metrics by seed."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

BASE = Path('/home/test/OpticNeuroFusion_Starter')
E01 = BASE / 'results/e01_cv'
OUT = BASE / 'results/e02_analysis'
OUT.mkdir(parents=True, exist_ok=True)
MODES = ('fundus', 'oct', 'fusion')
SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3, 4, 5)
REQUIRED_P = {'case_id','y_true','y_pred','p0','p1','p2'}
REQUIRED_M = {'patient_id','fundus_path','oct_path','label','split'}
records = []
known_cases = {}
split_counts = {}
fold_ids = {}

for seed in SEEDS:
    for fold in FOLDS:
        tag = f'fold{fold}_seed{seed}'
        mf = E01 / 'manifests' / f'{tag}.csv'
        assert mf.exists(), f'Missing manifest: {mf}'
        m = pd.read_csv(mf, dtype={'patient_id':str})
        assert REQUIRED_M <= set(m.columns), f'Manifest columns: {tag}'
        m['patient_id'] = m['patient_id'].astype(str).str.strip().str.zfill(4)
        m['split'] = m['split'].astype(str).str.strip().str.lower()
        m['label'] = m['label'].astype(int)
        assert m['patient_id'].is_unique, f'Duplicate IDs in manifest: {tag}'
        assert set(m['label']).issubset({0,1,2}), f'Unexpected class: {tag}'
        split_counts[tag] = m['split'].value_counts().to_dict()
        indexed = m.set_index('patient_id')
        for r in m.itertuples(index=False):
            signature = (int(r.label), str(r.fundus_path), str(r.oct_path))
            if r.patient_id in known_cases:
                assert known_cases[r.patient_id] == signature, f'Label/path changed: {r.patient_id} ({tag})'
            else:
                known_cases[r.patient_id] = signature
        ids_by_mode = {}
        for mode in MODES:
            rd = E01 / 'runs' / f'{tag}_{mode}'
            for name in ('complete.json','metrics.json','predictions.csv'):
                assert (rd / name).is_file(), f'Missing {name}: {rd}'
            p = pd.read_csv(rd / 'predictions.csv', dtype={'case_id':str})
            assert REQUIRED_P <= set(p.columns), f'Prediction columns: {rd}'
            p['case_id'] = p['case_id'].astype(str).str.strip().str.zfill(4)
            assert p['case_id'].is_unique, f'Duplicate prediction cases: {rd}'
            ids = set(p['case_id'])
            assert ids <= set(indexed.index), f'Unknown case IDs: {rd}: {ids-set(indexed.index)}'
            joined = indexed.loc[p['case_id']]
            assert (p['y_true'].to_numpy(dtype=int) == joined['label'].to_numpy(dtype=int)).all(), f'Label mismatch: {rd}'
            these_splits = set(joined['split'])
            assert len(these_splits) == 1, f'Predictions mix splits: {rd}: {these_splits}'
            split = next(iter(these_splits))
            assert split not in ('train','training'), f'Predictions are on TRAIN data: {rd}'
            assert ids == set(m.loc[m['split']==split,'patient_id']), f'Incomplete evaluation split: {rd}'
            probs = p[['p0','p1','p2']].to_numpy(dtype=float)
            assert np.isfinite(probs).all(), f'Invalid probabilities: {rd}'
            assert ((probs >= -1e-6) & (probs <= 1+1e-6)).all(), f'Probability range: {rd}'
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-3), f'Probability sums: {rd}'
            assert (p['y_pred'].to_numpy(dtype=int) == probs.argmax(axis=1)).all(), f'y_pred / probs disagree: {rd}'
            metric = json.loads((rd / 'metrics.json').read_text())
            assert int(metric['n']) == len(p), f'Number mismatched in metrics: {rd}'
            p['fold'] = fold
            p['seed'] = seed
            p['mode'] = mode
            p['eval_split'] = split
            records.append(p)
            ids_by_mode[mode] = ids
        assert ids_by_mode['fundus'] == ids_by_mode['oct'] == ids_by_mode['fusion'], f'Modes evaluate different cases: {tag}'
        fold_ids[(seed,fold)] = ids_by_mode['fundus']

all_ids = set(known_cases)
for seed in SEEDS:
    seq = [case for fold in FOLDS for case in fold_ids[(seed,fold)]]
    assert len(seq) == len(set(seq)) == len(all_ids), f'Fold overlap or gaps: seed{seed}'
    assert set(seq) == all_ids, f'Not all cases covered: seed{seed}'

# Identical image path reused across different IDs is a warning of possible grouping leakage.
path_owners = defaultdict(set)
for cid, (_,fundus_path,oct_path) in known_cases.items():
    path_owners[fundus_path].add(cid)
    path_owners[oct_path].add(cid)
shared_paths = {p:sorted(v) for p,v in path_owners.items() if len(v)>1}

# Hash each unique source image to detect byte-identical images assigned different IDs.
digest_owners = defaultdict(set)
missing_images = []
for cid, (_,fundus_path,oct_path) in known_cases.items():
    for kind,path in (('fundus',fundus_path),('oct',oct_path)):
        fp = Path(path)
        if not fp.is_file():
            missing_images.append(path)
            continue
        h = hashlib.sha256()
        with fp.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024), b''):
                h.update(chunk)
        digest_owners[(kind,h.hexdigest())].add(cid)
shared_hashes = [{'modality':k[0], 'sha256':k[1], 'case_ids':sorted(v)}
                 for k,v in digest_owners.items() if len(v)>1]

all_pred = pd.concat(records, ignore_index=True)
all_pred = all_pred.sort_values(['seed','fold','mode','case_id'])
all_pred.to_csv(OUT/'e02_oof_predictions.csv', index=False)

seed_rows = []
for (seed,mode), sub in all_pred.groupby(['seed','mode'], sort=True):
    assert len(sub) == len(all_ids) and sub['case_id'].is_unique, f'OOF coverage: {seed} / {mode}'
    y = sub['y_true'].to_numpy(dtype=int)
    probs = sub[['p0','p1','p2']].to_numpy(dtype=float)
    pred = sub['y_pred'].to_numpy(dtype=int)
    seed_rows.append({'seed':int(seed),'mode':mode,'n':len(sub),
        'macro_auc_ovr':roc_auc_score(y,probs,multi_class='ovr',average='macro',labels=[0,1,2]),
        'macro_f1':f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),
        'balanced_accuracy':balanced_accuracy_score(y,pred)})
seed_df = pd.DataFrame(seed_rows)
seed_df.to_csv(OUT/'e02_seed_oof_metrics.csv', index=False)
report = {'completed_runs':len(records),'total_cases':len(all_ids),
          'split_counts':split_counts,'shared_paths':shared_paths,
          'identical_image_hashes_between_cases':shared_hashes,
          'missing_images':missing_images,
          'caveat':'Case-level ID/path/hash audit does NOT prove distinct patient identities.'}
(OUT/'e02_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

print('\n=== E02 STRUCTURAL AUDIT ===')
print('Completed runs:',len(records),'/45; unique case IDs:',len(all_ids))
print('Example manifest splits:',split_counts['fold1_seed42'])
print('Prediction splits:', sorted(all_pred['eval_split'].unique()))
print('Shared image paths across case IDs:',len(shared_paths))
print('Byte-identical images across case IDs:',len(shared_hashes))
print('Missing image files:',len(missing_images))
print('\n=== PER-SEED POOLED OOF METRICS ===')
print(seed_df.round(4).to_string(index=False))
print('\n=== MEAN OF 3 SEED-LEVEL OOF RESULTS ===')
print(seed_df.groupby('mode')[['macro_auc_ovr','macro_f1','balanced_accuracy']].agg(['mean','std']).round(4).to_string())
print('\n=== PAIRED SEED-LEVEL AUROC DIFFERENCES ===')
pv = seed_df.pivot(index='seed',columns='mode',values='macro_auc_ovr')
for base in ('fundus','oct'):
    delta = pv['fusion']-pv[base]
    print(f'fusion - {base}:',', '.join(f'{v:+.4f}' for v in delta),'mean',f'{delta.mean():+.4f}')
print('\nSaved to:',OUT)
print('NOTE: patient_id may be a case ID; this audit cannot prove patient-level independence.')
