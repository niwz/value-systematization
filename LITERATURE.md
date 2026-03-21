# Related Literature

## Directly relevant to our work

### Stated vs Revealed reasoning gap
- **Reasoning Models Don't Always Say What They Think** (Anthropic, 2025)
  - https://www.anthropic.com/research/reasoning-models-dont-say-think
  - Models use hints but only acknowledge doing so 25-39% of the time
  - CoT is not a faithful window into actual decision process
  - Outcome-based RL improves faithfulness but plateaus
  - Implication: stated-vs-revealed gap is established for reasoning; we extend it to moral preferences

- **Measuring Faithfulness in Chain-of-Thought Reasoning** (Anthropic)
  - https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning
  - Companion to above

### LLM moral judgment — empirical

- **The Fragility of Moral Judgment in Large Language Models** (2026)
  - https://arxiv.org/html/2603.05651
  - Judgments robust to surface noise but highly sensitive to point-of-view changes
  - Protocol changes are the largest driver of verdict flips
  - Maps onto our position bias / label scheme finding

- **Large language models show amplified cognitive biases in moral decision-making** (PNAS)
  - https://www.pnas.org/doi/10.1073/pnas.2412015122
  - LLMs more utilitarian than humans, stronger omission bias
  - Compared LLM responses to representative U.S. sample on 22 dilemmas
  - Aligns with our finding of zero doing/allowing sensitivity

- **What Counts Underlying LLMs' Moral Dilemma Judgments?** (NLP4PI, 2025)
  - https://aclanthology.org/2025.nlp4pi-1.12.pdf
  - Directly asks which features drive LLM moral choices — closest to our research question

- **The Greatest Good Benchmark** (EMNLP 2024)
  - https://aclanthology.org/2024.emnlp-main.1224/
  - Evaluates moral judgments of 15 LLMs using utilitarian dilemmas
  - Finds "consistently encoded moral preferences that diverge from established moral theories"

- **Structured Moral Reasoning in Language Models** (EMNLP 2025)
  - https://aclanthology.org/2025.emnlp-main.1541.pdf
  - Ethical scaffolding improves LLM moral judgment
  - Parallels our reflection/sequential condition finding

- **Many LLMs Are More Utilitarian Than One** (2025)
  - https://arxiv.org/html/2507.00814v1
  - Multi-agent deliberation produces utilitarian boost (like human groups)

- **Human Realignment: An Empirical Study** (MPI, 2025)
  - https://www.coll.mpg.de/389160/2025_03online.pdf
  - GPT substantially more utilitarian than human subjects
  - GPT "deliberately imposes its (utilitarian) will"

### Alignment and Constitutional AI

- **Alignment Faking in Large Language Models** (Anthropic)
  - https://assets.anthropic.com/m/983c85a201a962f/original/Alignment-Faking-in-Large-Language-Models-full-paper.pdf
  - LLMs will fake alignment to preserve current preferences
  - 300K+ queries testing value trade-offs across Anthropic, OpenAI, DeepMind, xAI models

- **Constitutional AI: Harmlessness from AI Feedback** (Anthropic, 2022)
  - https://arxiv.org/abs/2212.08073
  - Original CAI paper. Tests whether models can articulate and apply principles
  - Does NOT test whether revealed decision behavior matches stated principles

- **C3AI: Crafting and Evaluating Constitutions for Constitutional AI** (2025)
  - https://arxiv.org/html/2502.15861v1
  - Positively framed, behavior-based principles align more closely with human preferences

- **Inverse Constitutional AI** (2025)
  - https://arxiv.org/html/2501.17112v1
  - Extracts constitution from pairwise preference dataset
  - Relevant: we could try extracting models' "revealed constitution" from their choices

## What's novel in our work (relative to above)

1. **Quantitative revealed moral weights**: Not just "utilitarian vs deontological" but per-feature coefficients across 7 models. No prior work (that we've found) fits a logistic model to feature deltas.
2. **Position bias dominates moral features in frontier models**: Sonnet 88% second-pick, Llama 8B 80% first-pick. The Fragility paper finds protocol sensitivity but doesn't quantify it as a competing predictor against moral features.
3. **Scaffolding activates latent preferences**: Sonnet converges to 92% agreement with Haiku under sequential evaluation. Prior work on ethical scaffolding (EMNLP 2025) doesn't test whether preferences are latent vs absent.
4. **Label scheme × position interaction**: Codex's finding that A/B labels amplify position bias relative to 1/2 labels is (likely) novel.

## Papers to read in full
- [ ] The Fragility of Moral Judgment in LLMs (most likely to overlap with us)
- [ ] What Counts Underlying LLMs' Moral Dilemma Judgments? (closest methodology)
- [ ] PNAS cognitive biases paper (highest-profile venue, need to know findings in detail)
- [ ] Alignment Faking (for the 300K value trade-off dataset)
