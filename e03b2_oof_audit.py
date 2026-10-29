from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
import e03b1_slice_policy as b

ROOT = Path('results/e03_b1')
OUT = ROOT / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44)
METHODS = ('uniform8', 'random8', 'quality8', 'diverse8')
PROB = ['p0', 'p1', 'p2']

ref = b.get_rows().set_index('patient_id')['label'].astype(int)
ids = sorted(ref.index)
y = ref.loc[ids].to_numpy()

assert len(ids) == 99 and set(y) == {0, 1, 2}

P, metrics = {}, []

for seed in SEEDS:
    for method in METHODS:
        batches = []

        for fold in range(1, 6):
            part = b.old.manifest(fold, seed)
            test_ids = set(
                part.loc[
                    part['split'] == 'test',
                    'patient_id'
                ].astype(str).str.zfill(4)
            )

            folder = ROOT / 'runs' / (
                f'fold{fold}_seed{seed}_{method}_gated'
            )

            if not (folder / 'complete.json').is_file():
                raise RuntimeError(f'Incomplete: {folder}')

            df = pd.read_csv(
                folder / 'predictions.csv',
                dtype={'case_id': str}
            )

            df['case_id'] = df['case_id'].str.zfill(4)

            assert df.case_id.is_unique
            assert set(df.case_id) == test_ids
            assert np.isfinite(df[PROB].to_numpy()).all()
            assert np.allclose(
                df[PROB].sum(axis=1), 1, atol=1e-4
            )

            assert all(
                int(r.y_true) == int(ref.loc[r.case_id])
                for r in df.itertuples()
            )

            batches.append(df)

        all_df = pd.concat(batches, ignore_index=True)

        assert len(all_df) == 99
        assert all_df.case_id.is_unique
        assert set(all_df.case_id) == set(ids)

        p = all_df.set_index('case_id').loc[
            ids, PROB
        ].to_numpy(float)

        P[method, seed] = p
        yh = p.argmax(axis=1)

        metrics.append({
            'method': method,
            'seed': seed,
            'macro_auc': roc_auc_score(
                y, p, multi_class='ovr',
                average='macro', labels=[0, 1, 2]
            ),
            'macro_f1': f1_score(
                y, yh, labels=[0, 1, 2],
                average='macro', zero_division=0
            ),
            'balanced_accuracy':
                balanced_accuracy_score(y, yh)
        })

m = pd.DataFrame(metrics)
m.to_csv(OUT / 'seed_oof_metrics.csv', index=False)

print('OOF CHECK PASS: 99 cases x 3 seeds x 4 methods')

print('\n=== POOLED OOF SUMMARY ===')
print(
    m.groupby('method')[
        ['macro_auc', 'macro_f1', 'balanced_accuracy']
    ].agg(['mean', 'std']).round(4)
)

def auc(yy, pp):
    values = []

    for c in (0, 1, 2):
        pos = yy == c
        npos = int(pos.sum())
        nneg = len(yy) - npos

        ranks = rankdata(
            pp[:, c], method='average'
        )

        score = (
            ranks[pos].sum()
            - npos * (npos + 1) / 2
        ) / (npos * nneg)

        values.append(score)

    return float(np.mean(values))

def avg_auc(method, ix):
    return float(np.mean([
        auc(y[ix], P[method, seed][ix])
        for seed in SEEDS
    ]))

rng = np.random.default_rng(20261026)
strata = [
    np.flatnonzero(y == k)
    for k in (0, 1, 2)
]

B = 2000
comparisons = ('random8', 'quality8', 'diverse8')
boot = {method: [] for method in comparisons}
full = np.arange(len(y))

print('\nBOOTSTRAP 2000...', flush=True)

for _ in range(B):
    ix = np.concatenate([
        rng.choice(s, len(s), replace=True)
        for s in strata
    ])

    baseline = avg_auc('uniform8', ix)

    for method in comparisons:
        boot[method].append(
            avg_auc(method, ix) - baseline
        )

rows = []

print('\n=== PAIRED BOOTSTRAP 95% CI ===')

for method in comparisons:
    delta = (
        avg_auc(method, full)
        - avg_auc('uniform8', full)
    )

    lo, hi = np.percentile(
        boot[method], [2.5, 97.5]
    )

    print(
        f'{method}: delta={delta:+.4f}, '
        f'95% CI=[{lo:+.4f}, {hi:+.4f}]',
        flush=True
    )

    rows.append({
        'comparison': f'{method}_minus_uniform8',
        'delta': delta,
        'ci_low': lo,
        'ci_high': hi,
        'bootstrap_samples': B
    })

pd.DataFrame(rows).to_csv(
    OUT / 'paired_bootstrap.csv',
    index=False
)

print('\nE03-B2 OOF AUDIT COMPLETE')
print('Saved:', OUT)
