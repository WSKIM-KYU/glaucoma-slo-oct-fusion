from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

base = Path('/home/test/OpticNeuroFusion_Starter')
runs = base / 'results/e03_a/runs'
manifests = base / 'results/e01_dedup/manifests'
out = base / 'results/e03_a/analysis'
out.mkdir(parents=True, exist_ok=True)

seeds = (42, 43, 44)
configs = (
    (8, 'gated'), (256, 'gated'),
    (8, 'concat'), (8, 'oct_attn'),
    (256, 'oct_attn')
)
pairs = (
    ((256, 'gated'), (8, 'gated')),
    ((256, 'gated'), (256, 'oct_attn')),
    ((8, 'gated'), (8, 'oct_attn')),
    ((8, 'concat'), (8, 'gated'))
)
cols = ['p0', 'p1', 'p2']

ref = pd.read_csv(
    manifests / 'fold1_seed42.csv',
    dtype={'patient_id': str}
)
ref['patient_id'] = ref['patient_id'].str.zfill(4)

assert len(ref) == 99 and ref['patient_id'].is_unique
assert '0077' not in set(ref['patient_id'])

labels = ref.set_index('patient_id')['label'].astype(int)
ids = sorted(labels.index)
y = labels.loc[ids].to_numpy()
assert set(y) == {0, 1, 2}

P = {}
rows = []

for seed in seeds:
    for n, mode in configs:
        frames = []

        for fold in range(1, 6):
            name = f'fold{fold}_seed{seed}'

            manifest = pd.read_csv(
                manifests / f'{name}.csv',
                dtype={'patient_id': str}
            )
            manifest['patient_id'] = (
                manifest['patient_id'].str.zfill(4)
            )

            expected = set(
                manifest.loc[
                    manifest['split'] == 'test',
                    'patient_id'
                ]
            )

            folder = runs / f'{name}_n{n}_{mode}'
            assert (folder / 'complete.json').is_file(), folder

            pred = pd.read_csv(
                folder / 'predictions.csv',
                dtype={'case_id': str}
            )
            pred['case_id'] = pred['case_id'].str.zfill(4)

            assert pred['case_id'].is_unique
            assert set(pred['case_id']) == expected, folder
            assert np.isfinite(pred[cols].to_numpy()).all()
            assert np.allclose(
                pred[cols].sum(axis=1), 1, atol=1e-4
            )

            assert all(
                int(r.y_true) == labels.loc[r.case_id]
                for r in pred.itertuples()
            )
            frames.append(pred)

        pooled = pd.concat(frames, ignore_index=True)

        assert len(pooled) == 99
        assert pooled['case_id'].is_unique
        assert set(pooled['case_id']) == set(ids)

        p = pooled.set_index('case_id').loc[
            ids, cols
        ].to_numpy(float)

        P[seed, n, mode] = p
        pred_class = p.argmax(axis=1)

        rows.append({
            'seed': seed,
            'slices': n,
            'mode': mode,
            'macro_auc': roc_auc_score(
                y, p, multi_class='ovr',
                average='macro', labels=[0, 1, 2]
            ),
            'macro_f1': f1_score(
                y, pred_class,
                labels=[0, 1, 2],
                average='macro',
                zero_division=0
            ),
            'balanced_accuracy':
                balanced_accuracy_score(y, pred_class)
        })

metrics = pd.DataFrame(rows)
metrics.to_csv(
    out / 'selected_seed_oof_metrics.csv',
    index=False
)

print('OOF COVERAGE PASS: 99 cases per seed/config')
print('\n=== OOF SUMMARY ===')
print(
    metrics.groupby(['slices', 'mode'])[
        ['macro_auc', 'macro_f1', 'balanced_accuracy']
    ].agg(['mean', 'std']).round(4)
)

def auc(yy, pp):
    values = []
    for cls in range(3):
        pos = yy == cls
        n1 = int(pos.sum())
        n0 = len(yy) - n1
        r = rankdata(pp[:, cls], method='average')
        values.append(
            (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        )
    return float(np.mean(values))

def mean_auc(cfg, ix):
    n, mode = cfg
    return np.mean([
        auc(y[ix], P[seed, n, mode][ix])
        for seed in seeds
    ])

rng = np.random.default_rng(20261026)
strata = [np.flatnonzero(y == c) for c in range(3)]
full = np.arange(len(y))
B = 2000
scores = {pair: [] for pair in pairs}

print('\nBOOTSTRAP 2000...', flush=True)

for i in range(B):
    ix = np.concatenate([
        rng.choice(s, len(s), replace=True)
        for s in strata
    ])

    v = {
        cfg: mean_auc(cfg, ix)
        for pair in pairs for cfg in pair
    }

    for pair in pairs:
        a, b = pair
        scores[pair].append(v[a] - v[b])

summary = []

print('\n=== EXPLORATORY PAIRED 95% CI ===')

for a, b in pairs:
    delta = mean_auc(a, full) - mean_auc(b, full)
    lo, hi = np.percentile(
        scores[a, b], [2.5, 97.5]
    )

    name = lambda cfg: f'n{cfg[0]}_{cfg[1]}'
    label = f'{name(a)}_minus_{name(b)}'

    print(
        label,
        f'delta={delta:+.4f}',
        f'CI=[{lo:+.4f},{hi:+.4f}]',
        flush=True
    )

    summary.append({
        'comparison': label,
        'delta': delta,
        'ci_low': lo,
        'ci_high': hi,
        'bootstrap_samples': B
    })

pd.DataFrame(summary).to_csv(
    out / 'selected_paired_bootstrap.csv',
    index=False
)

print('\nE03-A OOF AUDIT COMPLETE')
print('Saved:', out)
print('Exploratory case-level CI; patient identities unverified.')
