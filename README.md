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
```

### 3. Analyze results
```bash
python3 -m src.analysis
```

## Features

5 moral features: `benefit_magnitude`, `harm_magnitude`, `benefit_probability`, `directness_of_harm`, `beneficiary_identified`

1 nuisance feature: `option_order` (AB/BA)

## Conditions

- **No reflection**: bare system prompt
- **Domain reflection**: system prompt includes instruction to apply consistent principles

## Evaluation modes

- **Independent**: each item in a fresh API call
- **Sequential**: items sent in a running conversation with prior turns preserved
