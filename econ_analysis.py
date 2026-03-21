#!/usr/bin/env python3
"""Economics-oriented analysis of LLM moral decision-making data."""

import csv
import math
from collections import defaultdict
from pathlib import Path

base = Path("data/results")
models_dirs = ["haiku_v2", "sonnet_v2", "mistral_small_v2", "llama70b_v2", "llama8b_v2", "gpt4omini_v2", "gemma12b_v2"]
model_names = ["Haiku", "Sonnet", "Mistral 24B", "Llama 70B", "Llama 8B", "GPT-4o-mini", "Gemma 12B"]

features = ['delta_benefit_magnitude', 'delta_harm_magnitude', 'delta_benefit_probability',
            'delta_temporal_delay', 'delta_directness_of_harm', 'delta_beneficiary_identified',
            'delta_consent_of_harmed_party', 'delta_reversibility_of_harm']

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def parse_row(row):
    """Parse features and choice from a row."""
    chose_A = 1 if row['original_choice'] == 'A' else 0
    presented_first = 1 if row['presented_choice'] == 'A' else 0
    option_order = row['option_order']
    feats = {f: float(row[f]) for f in features}
    return chose_A, presented_first, option_order, feats

# Load all pre-reflection data
all_data = {}
for m, name in zip(models_dirs, model_names):
    rows = load_csv(base / m / "pre_choices.csv")
    parsed = [parse_row(r) for r in rows]
    all_data[name] = parsed

# Load all conditions
condition_files = {
    'pre': 'pre_choices.csv',
    'ind_none': 'post_independent_no_reflection.csv',
    'ind_domain': 'post_independent_domain_reflection.csv',
    'ind_prior': 'post_independent_prior_choice_reflection.csv',
    'seq_none': 'post_sequential_no_reflection.csv'
}

all_conditions = {}
for m, name in zip(models_dirs, model_names):
    for cond, fname in condition_files.items():
        fp = base / m / fname
        if fp.exists():
            rows = load_csv(fp)
            parsed = [parse_row(r) for r in rows]
            all_conditions[(name, cond)] = parsed

print("=" * 90)
print("ECONOMIST'S ANALYSIS OF LLM MORAL DECISION-MAKING")
print("=" * 90)

# ============================================================
# SECTION 1: Position Bias Analysis
# ============================================================
print("\n" + "=" * 90)
print("1. POSITION BIAS: PRESENTED-FIRST CHOICE RATES")
print("=" * 90)
print(f"\n{'Model':<15} {'Chose 1st':>10} {'N':>5} {'Rate':>8} {'When AB':>10} {'When BA':>10}")
print("-" * 65)

for name in model_names:
    data = all_data[name]
    total_first = sum(pf for _, pf, _, _ in data)
    n = len(data)

    # Split by order
    ab_first = sum(pf for _, pf, order, _ in data if order == 'AB')
    ab_n = sum(1 for _, _, order, _ in data if order == 'AB')
    ba_first = sum(pf for _, pf, order, _ in data if order == 'BA')
    ba_n = sum(1 for _, _, order, _ in data if order == 'BA')

    print(f"{name:<15} {total_first:>10} {n:>5} {total_first/n:>8.1%} "
          f"{ab_first/ab_n if ab_n else 0:>10.1%} {ba_first/ba_n if ba_n else 0:>10.1%}")

# Sonnet: check presented_choice = B rate
print("\nSONNET position bias detail:")
sonnet = all_data["Sonnet"]
chose_B_presented = sum(1 for r in sonnet if r[1] == 0)  # presented_choice != A
# Actually let's check raw: presented_choice column
sonnet_raw = load_csv(base / "sonnet_v2" / "pre_choices.csv")
b_count = sum(1 for r in sonnet_raw if r['presented_choice'] == 'B')
print(f"  Sonnet chose presented 'B': {b_count}/{len(sonnet_raw)} = {b_count/len(sonnet_raw):.1%}")
print(f"  Sonnet chose presented 'A': {len(sonnet_raw)-b_count}/{len(sonnet_raw)} = {(len(sonnet_raw)-b_count)/len(sonnet_raw):.1%}")

# ============================================================
# SECTION 2: Feature-by-Choice Crosstabs (manual logistic alternative)
# ============================================================
print("\n" + "=" * 90)
print("2. REVEALED PREFERENCES: CHOICE RATES BY FEATURE VALUES")
print("=" * 90)

# For the 5 linear models, compute P(choose A | feature direction)
linear_models = ["Haiku", "Llama 70B", "GPT-4o-mini", "Gemma 12B", "Mistral 24B"]

# For benefit_magnitude: group by sign of delta
print("\n--- P(Choose A) by sign of delta_benefit_magnitude ---")
print(f"{'Model':<15} {'neg (B>A)':>12} {'zero':>12} {'pos (A>B)':>12} {'N':>5}")
print("-" * 58)
for name in linear_models:
    data = all_data[name]
    groups = {'neg': [], 'zero': [], 'pos': []}
    for chose_a, _, _, feats in data:
        v = feats['delta_benefit_magnitude']
        if v < 0: groups['neg'].append(chose_a)
        elif v == 0: groups['zero'].append(chose_a)
        else: groups['pos'].append(chose_a)
    neg_rate = sum(groups['neg'])/len(groups['neg']) if groups['neg'] else float('nan')
    zero_rate = sum(groups['zero'])/len(groups['zero']) if groups['zero'] else float('nan')
    pos_rate = sum(groups['pos'])/len(groups['pos']) if groups['pos'] else float('nan')
    print(f"{name:<15} {neg_rate:>10.1%}({len(groups['neg']):>2}) "
          f"{zero_rate:>10.1%}({len(groups['zero']):>2}) "
          f"{pos_rate:>10.1%}({len(groups['pos']):>2}) {len(data):>5}")

# Same for harm_magnitude
print("\n--- P(Choose A) by sign of delta_harm_magnitude ---")
print("  (positive delta_harm = A has MORE harm than B)")
print(f"{'Model':<15} {'neg (A<harm)':>14} {'zero':>12} {'pos (A>harm)':>14} {'N':>5}")
print("-" * 62)
for name in linear_models:
    data = all_data[name]
    groups = {'neg': [], 'zero': [], 'pos': []}
    for chose_a, _, _, feats in data:
        v = feats['delta_harm_magnitude']
        if v < 0: groups['neg'].append(chose_a)
        elif v == 0: groups['zero'].append(chose_a)
        else: groups['pos'].append(chose_a)
    neg_rate = sum(groups['neg'])/len(groups['neg']) if groups['neg'] else float('nan')
    zero_rate = sum(groups['zero'])/len(groups['zero']) if groups['zero'] else float('nan')
    pos_rate = sum(groups['pos'])/len(groups['pos']) if groups['pos'] else float('nan')
    print(f"{name:<15} {neg_rate:>12.1%}({len(groups['neg']):>2}) "
          f"{zero_rate:>10.1%}({len(groups['zero']):>2}) "
          f"{pos_rate:>12.1%}({len(groups['pos']):>2}) {len(data):>5}")

# Same for benefit_probability
print("\n--- P(Choose A) by sign of delta_benefit_probability ---")
print(f"{'Model':<15} {'neg (B>A)':>12} {'zero':>12} {'pos (A>B)':>12}")
print("-" * 55)
for name in linear_models:
    data = all_data[name]
    groups = {'neg': [], 'zero': [], 'pos': []}
    for chose_a, _, _, feats in data:
        v = feats['delta_benefit_probability']
        if v < 0: groups['neg'].append(chose_a)
        elif v == 0: groups['zero'].append(chose_a)
        else: groups['pos'].append(chose_a)
    neg_rate = sum(groups['neg'])/len(groups['neg']) if groups['neg'] else float('nan')
    zero_rate = sum(groups['zero'])/len(groups['zero']) if groups['zero'] else float('nan')
    pos_rate = sum(groups['pos'])/len(groups['pos']) if groups['pos'] else float('nan')
    print(f"{name:<15} {neg_rate:>10.1%}({len(groups['neg']):>2}) "
          f"{zero_rate:>10.1%}({len(groups['zero']):>2}) "
          f"{pos_rate:>10.1%}({len(groups['pos']):>2})")

# ============================================================
# SECTION 3: Expected Utility Test
# ============================================================
print("\n" + "=" * 90)
print("3. EXPECTED UTILITY MAXIMIZER TEST")
print("   EU = benefit_magnitude × benefit_probability")
print("   If models are EU maximizers, they should choose the option with higher EU")
print("=" * 90)

print(f"\n{'Model':<15} {'EU-consistent':>14} {'N with EU diff':>15} {'Rate':>8}")
print("-" * 55)
for name in model_names:
    data = all_data[name]
    consistent = 0
    total = 0
    for chose_a, _, _, feats in data:
        # delta_EU = delta_benefit_mag * delta_benefit_prob is NOT right
        # We need EU_A - EU_B. But we only have deltas.
        # EU_A - EU_B approx relates to delta_benefit_mag and delta_benefit_prob
        # but they interact. Let's use a simpler test:
        # When benefit_mag and benefit_prob both favor same option, does model agree?
        bm = feats['delta_benefit_magnitude']
        bp = feats['delta_benefit_probability']

        # Both favor A (positive deltas)
        if bm > 0 and bp > 0:
            total += 1
            if chose_a == 1: consistent += 1
        # Both favor B (negative deltas)
        elif bm < 0 and bp < 0:
            total += 1
            if chose_a == 0: consistent += 1
        # Conflicting or zero: skip

    rate = consistent / total if total else float('nan')
    print(f"{name:<15} {consistent:>14} {total:>15} {rate:>8.1%}")

# More nuanced: when magnitude and probability conflict, which wins?
print("\n--- When benefit_magnitude and benefit_probability CONFLICT ---")
print("  (magnitude favors A, probability favors B, or vice versa)")
print(f"{'Model':<15} {'Chose mag':>10} {'Chose prob':>11} {'N':>5} {'Mag wins':>10}")
print("-" * 55)
for name in linear_models:
    data = all_data[name]
    mag_wins = 0
    prob_wins = 0
    total = 0
    for chose_a, _, _, feats in data:
        bm = feats['delta_benefit_magnitude']
        bp = feats['delta_benefit_probability']

        # Magnitude favors A, probability favors B
        if bm > 0 and bp < 0:
            total += 1
            if chose_a == 1: mag_wins += 1
            else: prob_wins += 1
        # Magnitude favors B, probability favors A
        elif bm < 0 and bp > 0:
            total += 1
            if chose_a == 0: mag_wins += 1
            else: prob_wins += 1

    if total:
        print(f"{name:<15} {mag_wins:>10} {prob_wins:>11} {total:>5} {mag_wins/total:>10.1%}")

# ============================================================
# SECTION 4: Loss Aversion / Harm Aversion
# ============================================================
print("\n" + "=" * 90)
print("4. LOSS/HARM AVERSION: ASYMMETRIC WEIGHTING OF HARM VS BENEFIT")
print("   Test: When harm_magnitude opposes benefit_magnitude, which dominates?")
print("=" * 90)

print(f"\n{'Model':<15} {'Harm-avoid':>11} {'Benefit-seek':>13} {'N':>5} {'Harm-avoid%':>12}")
print("-" * 60)
for name in linear_models:
    data = all_data[name]
    harm_avoid = 0
    benefit_seek = 0
    total = 0
    for chose_a, _, _, feats in data:
        bm = feats['delta_benefit_magnitude']
        hm = feats['delta_harm_magnitude']

        # A has more benefit AND more harm (bm > 0, hm > 0)
        if bm > 0 and hm > 0:
            total += 1
            if chose_a == 0: harm_avoid += 1
            else: benefit_seek += 1
        # B has more benefit AND more harm (bm < 0, hm < 0)
        elif bm < 0 and hm < 0:
            total += 1
            if chose_a == 1: harm_avoid += 1
            else: benefit_seek += 1

    if total:
        print(f"{name:<15} {harm_avoid:>11} {benefit_seek:>13} {total:>5} {harm_avoid/total:>12.1%}")

# ============================================================
# SECTION 5: Temporal Discounting
# ============================================================
print("\n" + "=" * 90)
print("5. TEMPORAL DISCOUNTING")
print("   Does temporal_delay affect choices? (positive delta = A is more delayed)")
print("=" * 90)

print(f"\n{'Model':<15} {'P(A|delay<0)':>13} {'P(A|delay=0)':>13} {'P(A|delay>0)':>13}")
print("-" * 58)
for name in linear_models:
    data = all_data[name]
    groups = {'neg': [], 'zero': [], 'pos': []}
    for chose_a, _, _, feats in data:
        v = feats['delta_temporal_delay']
        if v < 0: groups['neg'].append(chose_a)
        elif v == 0: groups['zero'].append(chose_a)
        else: groups['pos'].append(chose_a)
    neg_rate = sum(groups['neg'])/len(groups['neg']) if groups['neg'] else float('nan')
    zero_rate = sum(groups['zero'])/len(groups['zero']) if groups['zero'] else float('nan')
    pos_rate = sum(groups['pos'])/len(groups['pos']) if groups['pos'] else float('nan')
    print(f"{name:<15} {neg_rate:>13.1%} {zero_rate:>13.1%} {pos_rate:>13.1%}")

print("\n  (If models discount the future, P(A) should be LOWER when delay>0,")
print("   i.e., they should avoid the more delayed option)")

# ============================================================
# SECTION 6: Deontological Features - "Willingness to Pay"
# ============================================================
print("\n" + "=" * 90)
print("6. DEONTOLOGICAL FEATURE PREMIUMS")
print("   P(Choose A) conditional on binary features")
print("   Think of these as 'premiums' - how much does consent/identification shift choice?")
print("=" * 90)

binary_features = ['delta_consent_of_harmed_party', 'delta_reversibility_of_harm',
                   'delta_beneficiary_identified', 'delta_directness_of_harm']

for feat in binary_features:
    short = feat.replace('delta_', '')
    print(f"\n--- {short} ---")
    print(f"{'Model':<15} {'P(A|=-1)':>10} {'P(A|=0)':>10} {'P(A|=+1)':>10} {'Swing':>8}")
    print("-" * 55)
    for name in linear_models:
        data = all_data[name]
        groups = {-1: [], 0: [], 1: []}
        for chose_a, _, _, feats in data:
            v = int(feats[feat])
            groups[v].append(chose_a)

        rates = {}
        for k in [-1, 0, 1]:
            rates[k] = sum(groups[k])/len(groups[k]) if groups[k] else float('nan')

        swing = rates[1] - rates[-1] if not (math.isnan(rates[1]) or math.isnan(rates[-1])) else float('nan')
        print(f"{name:<15} {rates[-1]:>10.1%} {rates[0]:>10.1%} {rates[1]:>10.1%} {swing:>+8.1%}")

# ============================================================
# SECTION 7: Inter-model Agreement
# ============================================================
print("\n" + "=" * 90)
print("7. INTER-MODEL AGREEMENT (same items, pre-reflection)")
print("   Pairwise agreement rates on original_choice")
print("=" * 90)

# All models see same items in same order
print(f"\n{'':15}", end='')
for name in model_names:
    print(f"{name[:8]:>10}", end='')
print()
print("-" * 85)

for name_i in model_names:
    print(f"{name_i:<15}", end='')
    for name_j in model_names:
        data_i = all_data[name_i]
        data_j = all_data[name_j]
        agree = sum(1 for (a1, _, _, _), (a2, _, _, _) in zip(data_i, data_j) if a1 == a2)
        n = min(len(data_i), len(data_j))
        print(f"{agree/n:>10.1%}", end='')
    print()

# ============================================================
# SECTION 8: Sonnet's Second-Position Bias as Anchoring
# ============================================================
print("\n" + "=" * 90)
print("8. SONNET POSITION BIAS: RECENCY EFFECT ANALYSIS")
print("   Does Sonnet's B-bias vary by dilemma difficulty?")
print("=" * 90)

# Define "difficult" items as those where linear models disagree
# First, get consensus of linear models
consensus = []
for i in range(60):
    votes = sum(all_data[name][i][0] for name in linear_models)
    consensus.append(votes)  # 0-5, number choosing A

print(f"\n{'Consensus (# linear models choosing A)':40} {'Sonnet P(A)':>12} {'N':>5}")
print("-" * 60)
for threshold in range(6):
    items = [(i, all_data["Sonnet"][i][0]) for i in range(60) if consensus[i] == threshold]
    if items:
        rate = sum(chose_a for _, chose_a in items) / len(items)
        print(f"  {threshold} of 5 linear models chose A:                  {rate:>12.1%} {len(items):>5}")

# Does Sonnet agree with the linear consensus?
print(f"\nSonnet agreement with linear majority:")
agree = 0
for i in range(60):
    linear_majority = 1 if consensus[i] >= 3 else 0
    if all_data["Sonnet"][i][0] == linear_majority:
        agree += 1
print(f"  {agree}/60 = {agree/60:.1%}")

# Compare with Llama 8B
print(f"\nLlama 8B agreement with linear majority:")
agree8 = 0
for i in range(60):
    linear_majority = 1 if consensus[i] >= 3 else 0
    if all_data["Llama 8B"][i][0] == linear_majority:
        agree8 += 1
print(f"  {agree8}/60 = {agree8/60:.1%}")

# ============================================================
# SECTION 9: Condition effects on feature sensitivity
# ============================================================
print("\n" + "=" * 90)
print("9. CONDITION EFFECTS: DOES REFLECTION CHANGE FEATURE SENSITIVITY?")
print("   P(Choose high-benefit option) across conditions")
print("=" * 90)

for name in ["Haiku", "Sonnet", "GPT-4o-mini", "Llama 70B"]:
    print(f"\n  {name}:")
    print(f"  {'Condition':<25} {'P(A|ben>0)':>12} {'P(A|ben<0)':>12} {'Swing':>8} {'N':>5}")
    print(f"  {'-'*65}")
    for cond in ['pre', 'ind_none', 'ind_domain', 'ind_prior', 'seq_none']:
        key = (name, cond)
        if key not in all_conditions:
            continue
        data = all_conditions[key]
        pos = [ca for ca, _, _, f in data if f['delta_benefit_magnitude'] > 0]
        neg = [ca for ca, _, _, f in data if f['delta_benefit_magnitude'] < 0]
        if pos and neg:
            p_rate = sum(pos)/len(pos)
            n_rate = sum(neg)/len(neg)
            swing = p_rate - n_rate
            print(f"  {cond:<25} {p_rate:>12.1%} {n_rate:>12.1%} {swing:>+8.1%} {len(data):>5}")

# ============================================================
# SECTION 10: Simple hand-computed logistic-like score
# ============================================================
print("\n" + "=" * 90)
print("10. MANUAL WEIGHTED SCORING: WHICH FEATURES PREDICT CHOICES?")
print("    Correlation between each feature and P(chose_A)")
print("=" * 90)

def point_biserial(x_vals, y_binary):
    """Compute point-biserial correlation between continuous x and binary y."""
    n = len(x_vals)
    if n < 3: return 0.0
    y1 = [x for x, y in zip(x_vals, y_binary) if y == 1]
    y0 = [x for x, y in zip(x_vals, y_binary) if y == 0]
    if not y1 or not y0: return 0.0
    m1 = sum(y1) / len(y1)
    m0 = sum(y0) / len(y0)
    sx = sum(xi**2 for xi in x_vals) / n - (sum(x_vals)/n)**2
    if sx <= 0: return 0.0
    p = len(y1) / n
    return (m1 - m0) * math.sqrt(p * (1-p)) / math.sqrt(sx)

print(f"\n{'Feature':<30}", end='')
for name in linear_models:
    print(f"{name[:10]:>12}", end='')
print(f"{'Sonnet':>12}{'Llama 8B':>12}")
print("-" * (30 + 12*7))

for feat in features:
    short = feat.replace('delta_', '')
    print(f"{short:<30}", end='')
    for name in linear_models + ["Sonnet", "Llama 8B"]:
        data = all_data[name]
        x = [f[feat] for _, _, _, f in data]
        y = [ca for ca, _, _, _ in data]
        r = point_biserial(x, y)
        print(f"{r:>+12.3f}", end='')
    print()

# ============================================================
# SECTION 11: Position bias by condition (Sonnet)
# ============================================================
print("\n" + "=" * 90)
print("11. SONNET POSITION BIAS BY CONDITION")
print("    Does reflection reduce Sonnet's second-option anchoring?")
print("=" * 90)

for cond in ['pre', 'ind_none', 'ind_domain', 'ind_prior', 'seq_none']:
    key = ("Sonnet", cond)
    if key not in all_conditions:
        continue
    data = all_conditions[key]
    first_count = sum(pf for _, pf, _, _ in data)
    n = len(data)
    print(f"  {cond:<25} chose-first: {first_count}/{n} = {first_count/n:.1%}  "
          f"chose-second: {n-first_count}/{n} = {(n-first_count)/n:.1%}")

# Same for Llama 8B
print(f"\n  Llama 8B position bias by condition:")
for cond in ['pre', 'ind_none', 'ind_domain', 'ind_prior', 'seq_none']:
    key = ("Llama 8B", cond)
    if key not in all_conditions:
        continue
    data = all_conditions[key]
    first_count = sum(pf for _, pf, _, _ in data)
    n = len(data)
    print(f"  {cond:<25} chose-first: {first_count}/{n} = {first_count/n:.1%}  "
          f"chose-second: {n-first_count}/{n} = {(n-first_count)/n:.1%}")

# ============================================================
# SECTION 12: High-stakes vs low-stakes
# ============================================================
print("\n" + "=" * 90)
print("12. STAKES ANALYSIS: DOES DECISION QUALITY VARY WITH MAGNITUDE?")
print("    When |delta_benefit_magnitude| is large (990-1000), do models agree more?")
print("=" * 90)

for threshold_name, condition in [("High stakes (|delta_bm| >= 900)", lambda f: abs(f['delta_benefit_magnitude']) >= 900),
                                   ("Low stakes (|delta_bm| <= 90)", lambda f: abs(f['delta_benefit_magnitude']) <= 90),
                                   ("Zero stakes (delta_bm = 0)", lambda f: f['delta_benefit_magnitude'] == 0)]:
    print(f"\n  {threshold_name}:")
    for name in model_names:
        data = all_data[name]
        items = [(ca, feats) for ca, _, _, feats in data if condition(feats)]
        if items:
            # For high stakes with positive delta, should choose A
            pos = [ca for ca, f in items if f['delta_benefit_magnitude'] > 0]
            neg = [ca for ca, f in items if f['delta_benefit_magnitude'] < 0]
            zero = [ca for ca, f in items if f['delta_benefit_magnitude'] == 0]
            parts = []
            if pos: parts.append(f"A-favored:{sum(pos)}/{len(pos)}={sum(pos)/len(pos):.0%}")
            if neg: parts.append(f"B-favored:{len(neg)-sum(neg)}/{len(neg)}={1-sum(neg)/len(neg):.0%}")
            if zero: parts.append(f"neutral:{sum(zero)}/{len(zero)}")
            print(f"    {name:<15} N={len(items):>2}  {', '.join(parts)}")

print("\n" + "=" * 90)
print("END OF ANALYSIS")
print("=" * 90)
