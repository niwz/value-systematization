#!/usr/bin/env python3
"""Statistical analysis of LLM moral decision-making data."""

import csv
import os
import math
from collections import defaultdict, Counter
from pathlib import Path
import random

random.seed(42)

base = Path('/Users/nicwong/Desktop/value-systematization/data/results')

FEATURES = [
    'delta_benefit_magnitude', 'delta_harm_magnitude', 'delta_benefit_probability',
    'delta_temporal_delay', 'delta_directness_of_harm', 'delta_beneficiary_identified',
    'delta_consent_of_harmed_party', 'delta_reversibility_of_harm'
]

models_dirs = ['haiku_v2', 'sonnet_v2', 'mistral_small_v2', 'llama70b_v2', 'llama8b_v2', 'gpt4omini_v2', 'gemma12b_v2']
model_names = ['Haiku', 'Sonnet', 'Mistral 24B', 'Llama 70B', 'Llama 8B', 'GPT-4o-mini', 'Gemma 12B']


def chose_presented_A(row):
    return row['presented_choice'] == 'A'


def chose_presented_B(row):
    return row['presented_choice'] == 'B'


def get_label_scheme(row):
    return row.get('response_label_scheme', 'ab')


def chose_first_presented(row):
    scheme = get_label_scheme(row)
    if scheme == '12':
        return row['presented_choice'] == '1'
    return row['presented_choice'] == 'A'


def chose_second_presented(row):
    scheme = get_label_scheme(row)
    if scheme == '12':
        return row['presented_choice'] == '2'
    return row['presented_choice'] == 'B'

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_model_data(model_dir):
    all_rows = []
    d = base / model_dir
    for f in sorted(d.glob('*.csv')):
        if 'sanity' in f.name:
            continue
        all_rows.extend(load_csv(f))
    return all_rows

all_data = {}
for md, mn in zip(models_dirs, model_names):
    all_data[mn] = load_model_data(md)

print("="*80)
print("SECTION 1: DATA OVERVIEW")
print("="*80)
for name, rows in all_data.items():
    conditions = Counter(r['condition'] for r in rows)
    print(f"\n{name}: {len(rows)} total rows")
    for c, n in sorted(conditions.items()):
        print(f"  {c}: {n}")

# ============================================================================
print("\n" + "="*80)
print("SECTION 2: LABEL VS POSITION BIAS ANALYSIS")
print("="*80)

for name, rows in all_data.items():
    pre_rows = [r for r in rows if r['mode'] == 'pre']
    n = len(pre_rows)

    pres_a = sum(1 for r in pre_rows if r['presented_choice'] == 'A')
    pres_b = n - pres_a
    ab_count = sum(1 for r in pre_rows if r['option_order'] == 'AB')
    ba_count = n - ab_count
    orig_a = sum(1 for r in pre_rows if r['original_choice'] == 'A')
    orig_b = n - orig_a

    print(f"\n{name} (pre, n={n}):")
    print(f"  order: AB={ab_count}, BA={ba_count}")
    first_count = sum(1 for r in pre_rows if chose_first_presented(r))
    second_count = n - first_count

    print(f"  presented labels: A={pres_a} ({pres_a/n*100:.1f}%), B={pres_b} ({pres_b/n*100:.1f}%)")
    print(f"  presented position: first={first_count} ({first_count/n*100:.1f}%), second={second_count} ({second_count/n*100:.1f}%)")
    print(f"  original:  A={orig_a} ({orig_a/n*100:.1f}%), B={orig_b} ({orig_b/n*100:.1f}%)")

    # Separate label bias from true position bias
    ab_sub = [r for r in pre_rows if r['option_order'] == 'AB']
    ba_sub = [r for r in pre_rows if r['option_order'] == 'BA']
    if ab_sub:
        ab_pres_a = sum(1 for r in ab_sub if r['presented_choice'] == 'A')
        ab_first = sum(1 for r in ab_sub if chose_first_presented(r))
        print(f"  order=AB: pick label-A = {ab_pres_a}/{len(ab_sub)} ({ab_pres_a/len(ab_sub)*100:.1f}%)")
        print(f"           pick first = {ab_first}/{len(ab_sub)} ({ab_first/len(ab_sub)*100:.1f}%)")
    if ba_sub:
        ba_pres_a = sum(1 for r in ba_sub if r['presented_choice'] == 'A')
        ba_first = sum(1 for r in ba_sub if chose_first_presented(r))
        print(f"  order=BA: pick label-A = {ba_pres_a}/{len(ba_sub)} ({ba_pres_a/len(ba_sub)*100:.1f}%)")
        print(f"           pick first = {ba_first}/{len(ba_sub)} ({ba_first/len(ba_sub)*100:.1f}%)")

# ============================================================================
print("\n" + "="*80)
print("SECTION 3: DESIGN MATRIX PROPERTIES")
print("="*80)

pre_rows = [r for r in all_data['Haiku'] if r['mode'] == 'pre']
print(f"\nFeature stats (Haiku pre, n={len(pre_rows)}):")
for feat in FEATURES:
    vals = [float(r[feat]) for r in pre_rows]
    mean = sum(vals) / len(vals)
    var = sum((v - mean)**2 for v in vals) / len(vals)
    std = math.sqrt(var)
    unique = len(set(vals))
    print(f"  {feat.replace('delta_',''):30s}: mean={mean:8.2f} std={std:8.2f} unique={unique}")

# Correlations
print("\nPairwise correlations:")
def pearson(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    sx = math.sqrt(sum((xi-mx)**2 for xi in x)/n)
    sy = math.sqrt(sum((yi-my)**2 for yi in y)/n)
    if sx == 0 or sy == 0:
        return 0
    return sum((xi-mx)*(yi-my) for xi, yi in zip(x,y)) / (n*sx*sy)

feat_vals = {f: [float(r[f]) for r in pre_rows] for f in FEATURES}

print(f"{'':30s}", end="")
for f in FEATURES:
    print(f"{f.replace('delta_','')[:7]:>8s}", end="")
print()
for f1 in FEATURES:
    print(f"  {f1.replace('delta_',''):28s}", end="")
    for f2 in FEATURES:
        r = pearson(feat_vals[f1], feat_vals[f2])
        print(f"{r:8.3f}", end="")
    print()

# ============================================================================
print("\n" + "="*80)
print("SECTION 4: TEMPLATE FAMILY EFFECTS")
print("="*80)

for name in model_names:
    rows = all_data[name]
    pre_rows = [r for r in rows if r['mode'] == 'pre']
    templates = sorted(set(r['template_family'] for r in pre_rows))
    print(f"\n{name}:")
    for t in templates:
        sub = [r for r in pre_rows if r['template_family'] == t]
        n = len(sub)
        orig_a = sum(1 for r in sub if r['original_choice'] == 'A')
        pres_a = sum(1 for r in sub if chose_presented_A(r))
        first = sum(1 for r in sub if chose_first_presented(r))
        print(f"  {t:25s}: n={n:2d}, orig_A={orig_a:2d} ({orig_a/n*100:5.1f}%), label_A={pres_a:2d} ({pres_a/n*100:5.1f}%), first={first:2d} ({first/n*100:5.1f}%)")

# ============================================================================
print("\n" + "="*80)
print("SECTION 5: LOGISTIC REGRESSION")
print("="*80)

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def logistic_fit(X, y, lr=0.01, epochs=1000, l2=1.0):
    n, p = len(X), len(X[0])
    w = [0.0] * p
    b = 0.0
    for _ in range(epochs):
        dw = [0.0] * p
        db = 0.0
        for i in range(n):
            z = sum(w[j]*X[i][j] for j in range(p)) + b
            pred = sigmoid(z)
            err = pred - y[i]
            for j in range(p):
                dw[j] += err * X[i][j] / n
            db += err / n
        for j in range(p):
            w[j] -= lr * (dw[j] + l2 * w[j] / n)
        b -= lr * db
    return w, b

def logistic_predict(X, w, b):
    return [1 if sigmoid(sum(w[j]*X[i][j] for j in range(len(w))) + b) >= 0.5 else 0 for i in range(len(X))]

def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

def standardize(X):
    n, p = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n))/n for j in range(p)]
    stds = [math.sqrt(sum((X[i][j]-means[j])**2 for i in range(n))/n) for j in range(p)]
    stds = [s if s > 0 else 1 for s in stds]
    return [[(X[i][j]-means[j])/stds[j] for j in range(p)] for i in range(n)], means, stds

def standardize_with(X, means, stds):
    return [[(X[i][j]-means[j])/stds[j] for j in range(len(means))] for i in range(len(X))]

def kfold_cv(X, y, k=5, lr=0.05, epochs=2000, l2=1.0):
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    fold_size = n // k
    fold_accs = []
    for fold in range(k):
        test_idx = set(indices[fold*fold_size:(fold+1)*fold_size])
        train_idx = [i for i in range(n) if i not in test_idx]
        test_idx = list(test_idx)

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        X_train_s, means, stds = standardize(X_train)
        X_test_s = standardize_with(X_test, means, stds)
        w, b = logistic_fit(X_train_s, y_train, lr=lr, epochs=epochs, l2=l2)
        preds = logistic_predict(X_test_s, w, b)
        fold_accs.append(accuracy(y_test, preds))
    return fold_accs

print("\n5-fold CV accuracy (pre-reflection, features only):")
print(f"{'Model':15s} {'Mean':>8s} {'Folds':>50s} {'Std':>6s}")
for name in model_names:
    pre_rows = [r for r in all_data[name] if r['mode'] == 'pre']
    X = [[float(r[f]) for f in FEATURES] for r in pre_rows]
    y = [1 if r['original_choice'] == 'A' else 0 for r in pre_rows]
    fold_accs = kfold_cv(X, y)
    mean_acc = sum(fold_accs)/len(fold_accs)
    std_acc = math.sqrt(sum((a-mean_acc)**2 for a in fold_accs)/len(fold_accs))
    fold_str = ", ".join(f"{a:.3f}" for a in fold_accs)
    print(f"  {name:15s} {mean_acc:7.1%}   [{fold_str}]  {std_acc:.3f}")

print("\n5-fold CV accuracy (pre-reflection, features + position):")
for name in model_names:
    pre_rows = [r for r in all_data[name] if r['mode'] == 'pre']
    X = [[float(r[f]) for f in FEATURES] + [1.0 if r['option_order']=='BA' else 0.0] for r in pre_rows]
    y = [1 if r['original_choice'] == 'A' else 0 for r in pre_rows]
    fold_accs = kfold_cv(X, y)
    mean_acc = sum(fold_accs)/len(fold_accs)
    fold_str = ", ".join(f"{a:.3f}" for a in fold_accs)
    print(f"  {name:15s} {mean_acc:7.1%}   [{fold_str}]")

# ============================================================================
print("\n" + "="*80)
print("SECTION 6: BOOTSTRAP CIs ON ACCURACY")
print("="*80)

def bootstrap_cv(X, y, n_boot=200):
    n = len(X)
    accs = []
    for _ in range(n_boot):
        idx = [random.randint(0, n-1) for _ in range(n)]
        Xb = [X[i] for i in idx]
        yb = [y[i] for i in idx]
        fa = kfold_cv(Xb, yb, k=5, epochs=1500)
        accs.append(sum(fa)/len(fa))
    accs.sort()
    return accs[int(0.025*len(accs))], sum(accs)/len(accs), accs[int(0.975*len(accs))]

print(f"\n{'Model':15s} {'Mean':>8s} {'95% CI':>20s} {'Width':>8s}")
for name in model_names:
    pre_rows = [r for r in all_data[name] if r['mode'] == 'pre']
    X = [[float(r[f]) for f in FEATURES] for r in pre_rows]
    y = [1 if r['original_choice'] == 'A' else 0 for r in pre_rows]
    lo, mean, hi = bootstrap_cv(X, y, n_boot=200)
    print(f"  {name:15s} {mean:7.1%}   [{lo:.1%}, {hi:.1%}]   {(hi-lo)*100:5.1f}pp")

# ============================================================================
print("\n" + "="*80)
print("SECTION 7: POWER & CLASS BALANCE")
print("="*80)

for name in model_names:
    pre_rows = [r for r in all_data[name] if r['mode'] == 'pre']
    a_count = sum(1 for r in pre_rows if r['original_choice'] == 'A')
    b_count = len(pre_rows) - a_count
    minority = min(a_count, b_count)
    epv = minority / 8
    print(f"  {name:15s}: A={a_count}, B={b_count}, minority={minority}, EPV={epv:.1f}")

# ============================================================================
print("\n" + "="*80)
print("SECTION 8: STANDARDIZED COEFFICIENTS")
print("="*80)

print(f"\n{'Feature':30s}", end="")
for name in model_names:
    print(f"{name[:8]:>9s}", end="")
print()

for i, feat in enumerate(FEATURES):
    short = feat.replace('delta_','')
    print(f"  {short:28s}", end="")
    for name in model_names:
        pre_rows = [r for r in all_data[name] if r['mode'] == 'pre']
        X = [[float(r[f]) for f in FEATURES] for r in pre_rows]
        y = [1 if r['original_choice'] == 'A' else 0 for r in pre_rows]
        X_std, _, _ = standardize(X)
        w, b = logistic_fit(X_std, y, lr=0.05, epochs=3000, l2=1.0)
        print(f"{w[i]:9.3f}", end="")
    print()

# ============================================================================
print("\n" + "="*80)
print("SECTION 9: TEST-RETEST RELIABILITY")
print("="*80)

for name in model_names:
    rows = all_data[name]
    item_choices = defaultdict(dict)
    for r in rows:
        key = r['mode'] + '/' + r['condition'] + '/' + r['evaluation_mode']
        item_choices[r['item_id']][key] = r['original_choice']

    cond_list = sorted(set().union(*(d.keys() for d in item_choices.values())))
    print(f"\n{name}:")
    for i, c1 in enumerate(cond_list):
        for j, c2 in enumerate(cond_list):
            if j <= i:
                continue
            agree, total = 0, 0
            for iid, choices in item_choices.items():
                if c1 in choices and c2 in choices:
                    total += 1
                    if choices[c1] == choices[c2]:
                        agree += 1
            if total >= 10:
                print(f"  {c1:50s} vs {c2:50s}: {agree}/{total} ({agree/total*100:.0f}%)")

# ============================================================================
print("\n" + "="*80)
print("SECTION 10: SONNET LABEL VS POSITION BIAS")
print("="*80)

sonnet_pre = [r for r in all_data['Sonnet'] if r['mode'] == 'pre']
n = len(sonnet_pre)

# By template
print("\nSonnet bias by template family:")
for t in sorted(set(r['template_family'] for r in sonnet_pre)):
    sub = [r for r in sonnet_pre if r['template_family'] == t]
    ns = len(sub)
    label_b = sum(1 for r in sub if chose_presented_B(r))
    first = sum(1 for r in sub if chose_first_presented(r))
    second = sum(1 for r in sub if chose_second_presented(r))
    print(f"  {t:25s}: n={ns:2d}, label-B={label_b} ({label_b/ns*100:.1f}%), first={first} ({first/ns*100:.1f}%), second={second} ({second/ns*100:.1f}%)")

# By benefit_magnitude delta size
print("\nSonnet bias by |delta_benefit_magnitude| (above/below median):")
bm_vals = [abs(float(r['delta_benefit_magnitude'])) for r in sonnet_pre]
bm_median = sorted(bm_vals)[len(bm_vals)//2]
for label, condition in [("<=median", lambda r: abs(float(r['delta_benefit_magnitude'])) <= bm_median),
                          (">median", lambda r: abs(float(r['delta_benefit_magnitude'])) > bm_median)]:
    sub = [r for r in sonnet_pre if condition(r)]
    ns = len(sub)
    label_b = sum(1 for r in sub if chose_presented_B(r))
    first = sum(1 for r in sub if chose_first_presented(r))
    second = sum(1 for r in sub if chose_second_presented(r))
    print(f"  |delta_bm| {label:10s}: n={ns:2d}, label-B={label_b} ({label_b/ns*100:.1f}%), first={first} ({first/ns*100:.1f}%), second={second} ({second/ns*100:.1f}%)")

# Llama 8B similarly
print("\nLlama 8B bias by template family:")
llama8b_pre = [r for r in all_data['Llama 8B'] if r['mode'] == 'pre']
for t in sorted(set(r['template_family'] for r in llama8b_pre)):
    sub = [r for r in llama8b_pre if r['template_family'] == t]
    ns = len(sub)
    label_a = sum(1 for r in sub if chose_presented_A(r))
    first = sum(1 for r in sub if chose_first_presented(r))
    second = sum(1 for r in sub if chose_second_presented(r))
    print(f"  {t:25s}: n={ns:2d}, label-A={label_a} ({label_a/ns*100:.1f}%), first={first} ({first/ns*100:.1f}%), second={second} ({second/ns*100:.1f}%)")

# ============================================================================
print("\n" + "="*80)
print("SECTION 11: ITEM OVERLAP")
print("="*80)

for name in model_names:
    rows = all_data[name]
    pre_items = set(r['item_id'] for r in rows if r['mode'] == 'pre')
    post_items = set(r['item_id'] for r in rows if r['mode'] != 'pre')
    overlap = pre_items & post_items
    pre_only = pre_items - post_items
    print(f"  {name:15s}: pre={len(pre_items)}, post_unique={len(post_items)}, overlap={len(overlap)}, pre_only={len(pre_only)}")

# ============================================================================
print("\n" + "="*80)
print("SECTION 12: CROSS-CONDITION ACCURACY")
print("="*80)

for name in model_names:
    rows = all_data[name]
    conditions = sorted(set((r['mode'], r['condition'], r['evaluation_mode']) for r in rows))
    print(f"\n{name}:")
    for mode, cond, ev in conditions:
        sub = [r for r in rows if r['mode']==mode and r['condition']==cond and r['evaluation_mode']==ev]
        if len(sub) < 20:
            continue
        X = [[float(r[f]) for f in FEATURES] for r in sub]
        y = [1 if r['original_choice']=='A' else 0 for r in sub]
        fa = kfold_cv(X, y, k=5)
        ma = sum(fa)/len(fa)
        label_a = sum(1 for r in sub if chose_presented_A(r))/len(sub)*100
        first = sum(1 for r in sub if chose_first_presented(r))/len(sub)*100
        print(f"  {mode:20s} {cond:25s} n={len(sub):3d}  acc={ma:.1%}  label-A={label_a:.0f}%  first={first:.0f}%")

print("\n" + "="*80)
print("DONE")
print("="*80)
