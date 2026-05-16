# Poster Sign-Count CI Analysis

Wilson 95% intervals over sign-count comparisons. Treat these as descriptive intervals, not independent-observation causal inference: comparisons share prompts, models, and scaffolds.

| Domain | Contrast | Model | Count | Rate | Wilson 95% | Non-informative | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| capitalism | low_to_high_more_capitalist | Sonnet 4.5 | 3/16 | 18.8% | [6.6%, 43.0%] | 4 | Curated pro-market set: commercial rent, shortage pricing, AI labor, ticket auction, congestion pricing. Excludes public plaza, SaaS renewal, privatization prompts, and risk check. |
| capitalism | low_to_high_more_capitalist | GPT-5.4 | 13/20 | 65.0% | [43.3%, 81.9%] | 0 | Curated pro-market set: commercial rent, shortage pricing, AI labor, ticket auction, congestion pricing. Excludes public plaza, SaaS renewal, privatization prompts, and risk check. |
| capitalism | low_to_high_more_capitalist | GPT-5.5 | 14/20 | 70.0% | [48.1%, 85.5%] | 0 | Curated pro-market set: commercial rent, shortage pricing, AI labor, ticket auction, congestion pricing. Excludes public plaza, SaaS renewal, privatization prompts, and risk check. |
| generosity | low_to_high_more_generous | Sonnet 4.5 | 10/16 | 62.5% | [38.6%, 81.5%] | 0 | Four friendship-favor scenarios: moving help, stayover, airport ride, shortfall loan. |
| generosity | scaffold_vs_baseline_less_generous | Sonnet 4.5 | 24/24 | 100.0% | [86.2%, 100.0%] | 0 | Placebo/restate, deliberation, and rule-of-thumb compared with direct baseline at same model/scenario/effort. |
| generosity | low_to_high_more_generous | GPT-5.4 | 15/16 | 93.8% | [71.7%, 98.9%] | 0 | Four friendship-favor scenarios: moving help, stayover, airport ride, shortfall loan. |
| generosity | scaffold_vs_baseline_less_generous | GPT-5.4 | 24/24 | 100.0% | [86.2%, 100.0%] | 0 | Placebo/restate, deliberation, and rule-of-thumb compared with direct baseline at same model/scenario/effort. |
| generosity | low_to_high_more_generous | GPT-5.5 | 11/15 | 73.3% | [48.0%, 89.1%] | 1 | Four friendship-favor scenarios: moving help, stayover, airport ride, shortfall loan. |
| generosity | scaffold_vs_baseline_less_generous | GPT-5.5 | 24/24 | 100.0% | [86.2%, 100.0%] | 0 | Placebo/restate, deliberation, and rule-of-thumb compared with direct baseline at same model/scenario/effort. |

CSV: `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/runs/summaries/poster_sign_count_ci_20260515.csv`
Details: `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/runs/summaries/poster_sign_count_ci_20260515_details.csv`
