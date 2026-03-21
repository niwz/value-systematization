# Phase 2 Findings: Linear Compressibility of Moral Decision-Making Across 7 LLMs

## Setup

We presented moral dilemmas to 7 LLMs and measured how well a logistic regression on 8 moral features could predict each model's binary choices (A/B). Each dilemma is a pair of options parameterized by:

- **benefit_magnitude** {10, 100, 1000}
- **harm_magnitude** {0, 1, 10}
- **benefit_probability** {0.2, 0.5, 1.0}
- **temporal_delay** {0, 10, 100} years
- **directness_of_harm** {0, 1}
- **beneficiary_identified** {0, 1}
- **consent_of_harmed_party** {0, 1}
- **reversibility_of_harm** {0, 1}

The feature matrix uses deltas (Option A - Option B) for each feature. Three template families render features into natural language (rescue/triage, policy/prevention, direct harm tradeoff). Option order is randomized as a nuisance variable.

### Conditions (per model, 260 API calls)

| Condition | System prompt | Sees prior answers? | Evaluation |
|---|---|---|---|
| Pre-reflection (n=60) | Base | No | Independent |
| Independent, no reflection (n=50) | Base | No | Independent |
| Independent, domain reflection (n=50) | Generic moral principles | No | Independent |
| Independent, prior-choice reflection (n=50) | 10 prior dilemma-choice pairs | No | Independent |
| Sequential, no reflection (n=50) | Base | Yes (full history) | Sequential |

## Results

### Linear compressibility by model (pre-reflection, 5-fold CV)

| Model | Params | Vendor | Logistic | RF | Nonzero coefs |
|---|---|---|---|---|---|
| Haiku | ~20B | Anthropic | **88.3%** | 73.3% | 6/8 |
| Llama 3.3 | 70B | Meta | **85.0%** | 78.3% | 5/8 |
| GPT-4o-mini | ~8B | OpenAI | **83.3%** | 70.0% | 8/8 |
| Gemma 3 | 12B | Google | **76.7%** | 76.7% | 8/8 |
| Mistral Small | 24B | Mistral | **75.0%** | 70.0% | 4/8 |
| Llama 3.1 | 8B | Meta | **51.7%** | 55.0% | 4/8 |
| Sonnet | ~70B | Anthropic | **48.3%** | 50.0% | 0/8 |

### All conditions by model (logistic accuracy)

| Model | Pre | Ind. none | Ind. domain | Ind. prior | Sequential |
|---|---|---|---|---|---|
| Haiku | 88.3% | 88.0% | 86.0% | 86.0% | 72.0% |
| Llama 70B | 85.0% | 80.0% | 80.0% | 84.0% | 80.0% |
| GPT-4o-mini | 83.3% | 74.0% | 88.0% | 76.0% | 78.0% |
| Gemma 12B | 76.7% | 78.0% | 76.0% | 60.0% | 62.0% |
| Mistral 24B | 75.0% | 78.0% | 80.0% | 74.0% | 68.0% |
| Llama 8B | 51.7% | 48.0% | 52.0% | 60.0% | 56.0% |
| Sonnet | 48.3% | 68.0% | 62.0% | 74.0% | 70.0% |

## Key Findings

### 1. Most LLMs are linearly predictable in moral choices

Five of seven models achieve 75-88% cross-validated logistic accuracy from 8 moral features. A simple weighted sum of benefit magnitude, harm magnitude, and benefit probability predicts their choices well. This holds across vendors (Anthropic, Meta, OpenAI, Google, Mistral), sizes (8B-70B), and alignment approaches.

### 2. Model size does not predict linear compressibility

GPT-4o-mini (~8B) is 83.3% while Llama 3.1 8B is 51.7% — same size, completely different behavior. Llama 70B (85%) is more linear than Mistral 24B (75%). There is no monotonic relationship between parameter count and linearity.

### 3. Training quality matters more than size

The two models at chance (Llama 8B, Sonnet) are at opposite ends of the size spectrum. GPT-4o-mini at 8B achieves high linearity despite being the smallest model, likely because OpenAI's distillation gave it coherent value preferences. Llama 8B may lack the capacity for consistent moral preferences. The dividing line appears to be training quality, not parameter count.

### 4. Sonnet's original anomaly is largely a presentation artifact

Follow-up diagnostics show that Sonnet's low pre-reflection compressibility under the original `A/B` prompt is not well described as sophisticated non-linear moral reasoning. It is better described as a **strong bias toward the second presented option**, amplified by `A/B` response labels.

Evidence:
- **Original pre run**: Sonnet chose the second presented option on **53/60 (88.3%)** items
- **Matched AB/BA diagnostic, `A/B` labels**: on 30 paired base dilemmas, Sonnet flipped its underlying choice on **21/30** items when order reversed, and **all 21 flips** were in the direction implied by second-option bias (`AB:B -> BA:A`)
- **Matched AB/BA diagnostic, `1/2` labels**: the same directional order effect remained but weakened to **14/30** flips
- **Label ablation**: replacing `A/B` with `1/2` on the same pre items raised Sonnet's feature-logistic accuracy from **48.3%** to **70.0%**

So Sonnet is still order-sensitive under `1/2`, but `A/B` formatting makes the artifact substantially worse.

### 5. Reflection does not eliminate Sonnet's order bias in independent runs

For Sonnet's independent evaluations, the second-option bias persists under reflection:
- **Independent, no reflection**: second presented option on **45/50 (90%)**
- **Independent, domain reflection**: **45/50 (90%)**
- **Independent, prior-choice reflection**: **40/50 (80%)**

Prior-choice reflection improves compressibility for Sonnet, but it does **not** remove the underlying presentation bias. The most defensible interpretation is that reflection partly regularizes behavior while a strong order effect remains.

### 6. Sequential context changes Sonnet, but this is not a clean reflection comparison

In the sequential no-reflection run, Sonnet no longer shows the same strong second-option bias (**21/50 = 42%** second-option choices). However, this should not be overinterpreted:
- the sequential run uses a different item sample than pre/independent runs
- sequential context changes the task itself by preserving conversation history

So sequential context appears to attenuate the presentation artifact, but it is not a clean causal test of reflection alone.

### 7. Dominant features are consistent across models

For models in the linear tier, the top-weighted features are consistently:
1. **benefit_magnitude** or **benefit_probability** (positive)
2. **harm_magnitude** (negative)
3. **consent_of_harmed_party** (positive, where applicable)

This suggests a broadly consequentialist decision rule: maximize expected benefit, minimize harm, weight consent.

## Limitations

- **n=60 items** per condition is small for 8 features; coefficient estimates have wide confidence intervals
- **7 models** is insufficient for parametric claims about what drives linearity
- **Template confound**: Features are rendered into natural language; models may attend to surface phrasing rather than underlying features
- **Presentation-format confound**: Sonnet is highly sensitive to response labels and option order, so prompt mechanics can dominate apparent "moral" structure
- **Deterministic evaluation** (temperature=0): Results may differ under stochastic sampling
- **No frontier comparison** for Sonnet: We didn't test GPT-4o, Opus, or Gemini Pro to see if other frontier models share Sonnet's non-linearity

## Open Questions

1. **Is Sonnet's non-linearity a general property of frontier Anthropic models?** Testing Opus would clarify whether this is a Constitutional AI effect at scale.
2. **Is Llama 8B at chance because it's incapable or inconsistent?** Examining its raw responses for refusals, hedging, or random behavior would help distinguish capacity failure from genuine unpredictability.
3. **What features is Sonnet using?** Letting Sonnet explain its reasoning (max_tokens > 8) and coding the explanations could reveal dimensions outside our schema.
4. **Does the "midwit morality" curve exist?** With only one model in the "non-linear frontier" bucket, we can't confirm whether capability eventually curves back toward non-linearity.

## Cost

- Anthropic models (Haiku, Sonnet): ~$1-2 total
- OpenRouter models (Mistral, Llama 70B, Llama 8B, GPT-4o-mini, Gemma): ~$0.03 total
- Total: ~$2-3 for 1,890 API calls across 7 models
