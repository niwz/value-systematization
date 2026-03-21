# Quick Slides Outline

## Slide 1: Question and setup

**Title:** Reflective Value Systematization in LLMs

**Bullets:**
- Question: do simple reflection prompts make LLM moral choices more coherent or more compressible?
- Setup: 7 models, generated two-option moral dilemmas, 8 hand-coded moral features
- Main test: can a sparse logistic model predict each model's choices from those features?

**What to say:**
This pilot asks whether LLM moral choices can be summarized by a compact feature model, and whether reflection makes that structure clearer. We generated dilemmas from a fixed feature schema and treated compressibility under logistic regression as the main outcome.

## Slide 2: Experimental design

**Title:** Design

**Bullets:**
- 3 template families: rescue/triage, policy/prevention, direct-harm tradeoff
- 8 features: benefit magnitude, harm magnitude, probability, delay, directness, identified beneficiary, consent, reversibility
- Conditions per model:
  - pre-reflection: 60 items
  - independent no reflection: 50
  - independent domain reflection: 50
  - independent prior-choice reflection: 50
  - sequential no reflection: 50
- Deterministic runs (`temperature=0`), 260 calls per model

**What to say:**
The design is intentionally simple. Each dilemma is a pair of options defined by the same small set of moral features, and we compare baseline prompting to a few light-touch reflection variants.

## Slide 3: Main result

**Title:** Most Models Are Fairly Compressible

**Bullets:**
- Pre-reflection logistic accuracy:
  - Haiku: 0.883
  - Llama 70B: 0.850
  - GPT-4o-mini: 0.833
  - Gemma 12B: 0.767
  - Mistral Small: 0.750
  - Llama 8B: 0.517
- 5 of 7 models show strong linear structure from a small feature set
- Size is not the main story: GPT-4o-mini does well; Llama 8B does not

**What to say:**
The central result is that most models are much more predictable than I expected from a very small linear feature model. The exception on the low end is Llama 8B, which stays near chance.

## Slide 4: What the models seem to care about

**Title:** Shared Feature Structure

**Bullets:**
- Recurrent high-weight features across stronger models:
  - benefit magnitude: positive
  - benefit probability: positive
  - harm magnitude: negative
  - consent: often positive
- Rough interpretation: consequentialist-looking tradeoff with some role for consent

**What to say:**
The stronger models are not just predictable in arbitrary ways. They tend to load on the same interpretable dimensions: more expected benefit, less harm, and in some models more weight on consent.

## Slide 5: Reflection results

**Title:** Reflection Effects Are Mixed

**Bullets:**
- No uniform improvement from reflection across models
- Domain reflection sometimes helps:
  - GPT-4o-mini: 0.833 -> 0.880
  - Mistral Small: 0.750 -> 0.800
- Prior-choice reflection is often worse or unstable
- Sequential runs are generally lower for the stronger models

**What to say:**
At least in this pilot, reflection is not a clean win. The most favorable condition is often simple domain reflection, but the effect is model-specific and not consistently positive.

## Slide 6: Sonnet anomaly

**Title:** Important Caveat: Sonnet

**Bullets:**
- Original pre-reflection Sonnet result: 0.483 logistic accuracy
- Follow-up suggests strong presentation sensitivity rather than clean nonlinear moral structure
- Evidence:
  - second-presented option on 53/60 original pre items
  - paired AB/BA diagnostic: 21/30 flips under `A/B`
  - same effect weakens but remains under `1/2`
  - relabeling pre run to `1/2` raises accuracy to 0.700

**What to say:**
I would present Sonnet as an anomaly and a methodological warning, not as the headline result. Right now the safest interpretation is that Sonnet is unusually sensitive to response labels and option order in this battery.

## Slide 7: New follow-up

**Title:** New Follow-up: Ambiguity, Not Extremity

**Bullets:**
- Targeted 2x2 battery: low/high ambiguity x low/high extremity
- Sonnet paired-order flip rates:
  - corrected `A/B`: 43.8% low ambiguity vs 93.8% high ambiguity
  - corrected `1/2`: 25.0% low ambiguity vs 75.0% high ambiguity
- Extremity did not explain the effect
- Result replicated after fixing the 2x2 sampler
- Best current hypothesis: on low-consensus, low-signal, or conflict-heavy dilemmas, Sonnet falls back to order-sensitive responding

**What to say:**
This is the cleanest new result. After fixing a bug in the first 2x2 sampler and rerunning Sonnet, the pattern still held: order sensitivity tracked the ambiguity proxy strongly and did not track extremity. That makes the Sonnet issue look more like low-signal or conflict-conditioned positional fallback than a pure reaction to high-stakes or extreme cases.

## Slide 8: Takeaways

**Title:** Tentative Takeaways

**Bullets:**
- Many LLMs have moral-choice behavior that is surprisingly compressible by a simple feature model
- This pattern appears across vendors and is not explained by parameter count alone
- Light-touch reflection does not reliably improve coherence/compressibility
- Evaluation format matters: Sonnet shows that prompt mechanics can strongly distort apparent value structure, especially on low-margin / low-consensus items

**What to say:**
The main message is that compact predictive structure is real for most of these models, but measurement is fragile. The Sonnet follow-up makes that methodological point much sharper.

## Slide 9: Next step

**Title:** Immediate Next Step

**Bullets:**
- Keep the main cross-model result
- Treat Sonnet as a separate investigation
- Next experiment: disentangle ambiguity, order effects, and label effects with tighter paired controls

**What to say:**
For tomorrow, I would frame the next step as tightening the measurement protocol rather than making stronger psychological claims about any one model.
