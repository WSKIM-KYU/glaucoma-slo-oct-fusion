#!/usr/bin/env python3
"""E03-C: pre-specified 8-B-scan end-to-end fine-tuning on E01-Dedup splits.

Exploratory internal development only. Never select final models on GAMMA outer-test results.
Uses the pretrained ResNet18 and deterministic inference transforms from E03-A.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset

import e03a_volume_mil as old

ROOT = Path(__file__).resolve().parent
MANIFESTS = ROOT / 'results/e01_dedup/manifests'
OUTPUT = ROOT / 'results/e03_c'
SEEDS = (42, 43, 44)
CONFIGS = ('frozen_concat', 'octlast_concat', 'bothlast_concat',
           'bothlast_gated', 'bothlast_cross',
           'octlast_gated', 'octlast_cross')
N_SLICES = 8
POSITIONS = old.get_positions(N_SLICES)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def manifest(fold, seed):
    path = MANIFESTS / f'fold{fold}_seed{seed}.csv'
    df = pd.read_csv(path, dtype={'patient_id': str})
    df['patient_id'] = df['patient_id'].str.zfill(4)
    assert {'patient_id', 'fundus_path', 'oct_path', 'label', 'split'} <= set(df.columns)
    assert len(df) == 99 and df.patient_id.is_unique, path
    assert set(df.split) == {'train', 'val', 'test'}, path
    assert '0077' not in set(df.patient_id)
    assert set(df.label.astype(int)) == {0, 1, 2}
    return df


def abs_path(p):
    p = Path(str(p))
    if p.exists():
        return p
    q = ROOT / p
    if q.exists():
        return q
    raise FileNotFoundError(str(p))


def paths_for_case(row):
    fundus = abs_path(row.fundus_path)
    central = abs_path(row.oct_path)
    scans = sorted(central.parent.glob('*_image.jpg'),
                   key=lambda p: int(p.name.split('_')[0]))
    indices = [int(p.stem.split('_')[0]) for p in scans]
    if len(scans) != 256 or indices != list(range(indices[0], indices[0] + 256)):
        raise RuntimeError(f'OCT incomplete: {central.parent}; count={len(scans)}; first={indices[:3]}; last={indices[-3:]}')
    return fundus, [scans[int(i)] for i in POSITIONS]


def preflight():
    ref = manifest(1, 42)
    ids = set(ref.patient_id)
    labels = dict(zip(ref.patient_id, ref.label.astype(int)))
    for r in ref.itertuples(index=False):
        f, scans = paths_for_case(r)
        assert f.is_file() and len(scans) == 8
    for seed in SEEDS:
        seen = []
        for fold in range(1, 6):
            df = manifest(fold, seed)
            assert set(df.patient_id) == ids
            assert all(labels[r.patient_id] == int(r.label) for r in df.itertuples())
            seen.extend(df.loc[df.split == 'test', 'patient_id'].tolist())
        assert len(seen) == 99 and len(set(seen)) == 99
    print('E03-C PREFLIGHT PASS: 99 deduplicated cases, 15 fixed manifests, 8 B-scans', flush=True)
    print('WARNING: case_id is not a verified person ID; patient-level independence unverified')


class RawDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.samples = []
        self.transform = transform
        for r in dataframe.itertuples(index=False):
            fundus, scans = paths_for_case(r)
            self.samples.append((r.patient_id, int(r.label), fundus, scans))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cid, label, fundus, scans = self.samples[index]
        with Image.open(fundus) as im:
            f = self.transform(im.convert('RGB'))
        zs = []
        for path in scans:
            with Image.open(path) as im:
                zs.append(self.transform(im.convert('RGB')))
        return f, torch.stack(zs), label, cid


class FusionHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.oct_proj = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.25))
        self.fundus_proj = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.25))
        self.att = nn.Sequential(nn.Linear(513, 96), nn.Tanh(), nn.Linear(96, 1))
        self.gate = nn.Linear(256, 128)
        self.combine = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.25))
        self.cross_pos = nn.Linear(1, 128)
        self.cross_att = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(128, 3)

    def forward(self, fundus, oct_features):
        b, s, d = oct_features.shape
        pos = torch.as_tensor(POSITIONS, device=oct_features.device,
                              dtype=oct_features.dtype).view(1, s, 1)
        pos = (pos / 255.0 * 2.0 - 1.0).expand(b, -1, -1)
        f = self.fundus_proj(fundus)
        if self.mode == 'cross':
            tokens = self.oct_proj(oct_features) + self.cross_pos(pos)
            context, _ = self.cross_att(f[:, None, :], tokens, tokens, need_weights=False)
            z = self.combine(torch.cat([f, context[:, 0, :]], dim=-1))
        else:
            score = self.att(torch.cat([oct_features, pos], dim=-1)).squeeze(-1)
            pool = (torch.softmax(score, dim=1).unsqueeze(-1) * oct_features).sum(dim=1)
            o = self.oct_proj(pool)
            together = torch.cat([o, f], dim=-1)
            if self.mode == 'concat':
                z = self.combine(together)
            elif self.mode == 'gated':
                gate = torch.sigmoid(self.gate(together))
                z = gate * f + (1.0 - gate) * o
            else:
                raise ValueError(self.mode)
        return self.classifier(z)


class Model(nn.Module):
    def __init__(self, tune, mode):
        super().__init__()
        self.fundus_encoder, tf = old.weights_and_transforms()
        self.oct_encoder, _ = old.weights_and_transforms()
        for encoder in (self.fundus_encoder, self.oct_encoder):
            if not hasattr(encoder, 'layer4'):
                raise RuntimeError('E03-A pretrained encoder is not torchvision ResNet18 with layer4')
            for p in encoder.parameters():
                p.requires_grad_(False)
        if tune in ('octlast', 'bothlast'):
            for p in self.oct_encoder.layer4.parameters():
                p.requires_grad_(True)
        if tune == 'bothlast':
            for p in self.fundus_encoder.layer4.parameters():
                p.requires_grad_(True)
        self.head = FusionHead(mode)
        self.transform = tf

    def train(self, mode=True):
        super().train(mode)
        # All pretrained BN running statistics are fixed on a 99-case dataset.
        self.fundus_encoder.eval()
        self.oct_encoder.eval()
        return self

    def forward(self, f, octs):
        b, s, c, h, w = octs.shape
        ff = self.fundus_encoder(f)
        if ff.ndim != 2 or ff.shape[1] != 512:
            raise RuntimeError(f'Unexpected fundus output {tuple(ff.shape)}')
        # Small chunks bound peak convolution activation memory on RTX 3090.
        zz = []
        for i in range(0, s, 4):
            x = octs[:, i:i+4].reshape(-1, c, h, w)
            z = self.oct_encoder(x)
            if z.ndim != 2 or z.shape[1] != 512:
                raise RuntimeError(f'Unexpected OCT output {tuple(z.shape)}')
            zz.append(z.reshape(b, -1, 512))
        return self.head(ff, torch.cat(zz, dim=1))


def metrics(y, p):
    yh = np.argmax(p, axis=1)
    try:
        auc = float(roc_auc_score(y, p, labels=[0,1,2], multi_class='ovr', average='macro'))
    except ValueError:
        auc = None
    return {
        'macro_auc_ovr': auc,
        'macro_f1': float(f1_score(y, yh, labels=[0,1,2], average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, yh)),
        'accuracy': float(accuracy_score(y, yh)),
    }


def fit_one(config, fold, seed, epochs, batch, smoke):
    experiment = f'fold{fold}_seed{seed}_{config}'
    dest = OUTPUT / ('smoke' if smoke else 'runs') / experiment
    if (dest / 'complete.json').exists():
        print('SKIP', experiment, flush=True)
        return
    tune, mode = config.split('_', 1)
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(tune, mode).to(device)
    df = manifest(fold, seed)
    train_df = df[df.split == 'train']
    val_df = df[df.split == 'val']
    test_df = df[df.split == 'test']
    dl = DataLoader(RawDataset(train_df, model.transform), batch_size=batch,
                    shuffle=True, num_workers=0)
    dv = DataLoader(RawDataset(val_df, model.transform), batch_size=batch,
                    shuffle=False, num_workers=0)
    dt = DataLoader(RawDataset(test_df, model.transform), batch_size=batch,
                    shuffle=False, num_workers=0)
    enc_params = [p for name, p in model.named_parameters()
                  if p.requires_grad and not name.startswith('head.')]
    groups = [{'params': list(model.head.parameters()), 'lr': 3e-4}]
    if enc_params:
        groups.append({'params': enc_params, 'lr': 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-3)
    counts = np.bincount(train_df.label.astype(int).to_numpy(), minlength=3)
    weights = torch.tensor(len(train_df) / (3 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_loss, best_epoch, best_state, wait = float('inf'), 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for f, o, y, _ in dl:
            f, o, y = f.to(device), o.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(f, o), y)
            loss.backward()
            optimizer.step()
        model.eval()
        losses, ns = 0., 0
        with torch.inference_mode():
            for f, o, y, _ in dv:
                f, o, y = f.to(device), o.to(device), y.to(device)
                ll = float(criterion(model(f, o), y))
                losses += ll * len(y)
                ns += len(y)
        val_loss = losses / ns
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 8:
            break

    model.load_state_dict(best_state)
    model.eval()
    ids, yy, pp = [], [], []
    with torch.inference_mode():
        for f, o, y, case_id in dt:
            logits = model(f.to(device), o.to(device))
            pp.extend(torch.softmax(logits, dim=1).cpu().tolist())
            yy.extend(y.tolist())
            ids.extend(case_id)
    y = np.asarray(yy, dtype=int)
    p = np.asarray(pp, dtype=float)
    result = {
        'experiment': experiment, 'fold': fold, 'seed': seed,
        'config': config, 'n_slices': N_SLICES, 'n_train': len(train_df),
        'n_val': len(val_df), 'n_test': len(test_df),
        'best_epoch': best_epoch, 'val_loss': best_loss,
        'trainable_parameters': sum(t.numel() for t in model.parameters() if t.requires_grad),
        **metrics(y, p),
    }
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'case_id': c, 'y_true': int(y[i]),
                   'y_pred': int(p[i].argmax()),
                   **{f'p{j}': float(p[i, j]) for j in range(3)}}
                  for i, c in enumerate(ids)]).to_csv(dest / 'predictions.csv', index=False)
    torch.save(best_state, dest / 'best.pt')
    (dest / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (dest / 'complete.json').write_text(json.dumps({'complete': True, 'experiment': experiment}), encoding='utf-8')
    print('COMPLETE', experiment, 'AUROC', result['macro_auc_ovr'],
          'best_epoch', best_epoch, 'val_loss', round(best_loss, 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('preflight', 'train', 'summary'), required=True)
    ap.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS[:3]))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--limit-runs', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.stage == 'preflight':
        preflight()
        return
    if args.stage == 'summary':
        rows = []
        for p in (OUTPUT / 'runs').glob('*/metrics.json'):
            if (p.parent / 'complete.json').exists():
                rows.append(json.loads(p.read_text()))
        if not rows:
            print('NO COMPLETED E03-C RUNS')
            return
        df = pd.DataFrame(rows)
        print('Completed:', len(df))
        print(df.groupby('config')[['macro_auc_ovr','macro_f1','balanced_accuracy']]
              .agg(['count','mean','std']).round(4).to_string())
        df.to_csv(OUTPUT / 'all_metrics.csv', index=False)
        return
    if args.epochs < 15 and not args.smoke:
        ap.error('Short epochs require --smoke (keeps production output separate)')
    preflight()
    count = 0
    for fold in range(1, 6):
        for seed in SEEDS:
            for config in args.configs:
                fit_one(config, fold, seed, args.epochs, args.batch, args.smoke)
                count += 1
                if args.limit_runs and count >= args.limit_runs:
                    return


if __name__ == '__main__':
    main()
