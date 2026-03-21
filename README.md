# Reflective Value Systematization in LLMs — Pilot

Tests whether reflection makes an LLM's moral choices more compressible and coherent under a compact predictive model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn anthropic pyyaml
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

### 1. Generate design matrix
```bash
python3 -m src.design_matrix
```

### 2. Run experiments
```bash
# Sanity check (25 items, no reflection)
python3 -m src.api_runner sanity

# Pre-reflection (60 items)
python3 -m src.api_runner pre

# Post-reflection independent evaluation (30 items × 2 conditions)
python3 -m src.api_runner post_independent

# Post-reflection sequential evaluation (30 items × 2 conditions)
python3 -m src.api_runner post_sequential

# Or run everything
python3 -m src.api_runner all

# Label-ablation run while keeping the same underlying dilemmas
python3 -m src.api_runner pre --response-label-scheme 12

# Write a rerun into a separate directory
python3 -m src.api_runner pre --response-label-scheme 12 --results-dir data/results/sonnet_v2_labels12

# Paired-order Sonnet diagnostic: each base item is shown once as AB and once as BA,
# under both A/B and 1/2 response labels
python3 -m src.api_runner order_ablation --config configs/pilot_v2.yaml --model claude-sonnet-4-5-20250929 --results-dir data/results/sonnet_order_ablation

# Small non-moral positional-bias diagnostic for Sonnet
python3 -m src.api_runner nonmoral_order_ablation --config configs/pilot_v2.yaml --model claude-sonnet-4-5-20250929 --results-dir data/results/sonnet_nonmoral_bias
```

### 3. Analyze results
```bash
python3 -m src.analysis

# Analyze the paired-order diagnostic separately
python3 -m src.order_ablation_analysis --results-dir data/results/sonnet_order_ablation

# Analyze the small non-moral bias battery
python3 -m src.nonmoral_bias_analysis --results-dir data/results/sonnet_nonmoral_bias
```

## Features

5 moral features: `benefit_magnitude`, `harm_magnitude`, `benefit_probability`, `directness_of_harm`, `beneficiary_identified`

1 nuisance feature: `option_order` (AB/BA)

## Conditions

- **No reflection**: bare system prompt
- **Domain reflection**: system prompt includes instruction to apply consistent principles

## Response label schemes

- **`ab`**: default; identical to the original experiment and preserves exact reproducibility
- **`12`**: replaces the presented response labels with `1/2` to test whether apparent effects are driven by `A/B` label bias rather than the dilemma features

## Sonnet-specific diagnostic

- **`order_ablation`**: a separate paired-order test intended for Sonnet debugging
- Each sampled base dilemma is duplicated with both `AB` and `BA` presentation order
- The mode runs both response-label schemes (`ab` and `12`) and writes separate files
- It does not change the default `pre`/`post` workflows or their output format

## General-bias diagnostic

- **`nonmoral_order_ablation`**: a tiny non-moral paired-order battery for testing whether Sonnet's presentation bias generalizes beyond moral dilemmas
- Uses fixed everyday forced-choice prompts and writes separate `ab` and `12` result files

## Evaluation modes

- **Independent**: each item in a fresh API call
- **Sequential**: items sent in a running conversation with prior turns preserved
