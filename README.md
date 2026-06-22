# TakeMeter — NBA Discourse Classifier

A fine-tuned DistilBERT text classifier that categorizes r/nba posts into three discourse types: **analysis**, **hot_take**, and **reaction**. Built as part of AI201 (Applications of AI Engineering), Project 3.

---

## Community

**Community:** r/nba (Reddit)

r/nba is one of the largest sports communities on the internet, with over 9 million members generating thousands of posts and comments on any given game day. It was chosen because NBA basketball naturally produces three distinct and roughly equal modes of written expression: rigorous analytical discourse citing statistics and advanced metrics, opinionated argumentation about players and franchises, and real-time emotional reactions to live events. These three modes are superficially similar — they all use basketball vocabulary and appear on the same platform — but they differ sharply in purpose, structure, and the social response they invite, which makes surface-level keyword matching insufficient and requires genuine semantic understanding.

---

## Label Taxonomy

Three mutually exclusive labels, distinguished by the author's primary purpose:

**`analysis`** — The author's primary purpose is to explain something about basketball using evidence: statistics, advanced metrics, film observations, lineup data, or tactical breakdowns. The evidence leads; any opinion is incidental.

- *Example 1:* "Their defensive rating improved 8 points since the trade deadline, driven primarily by better transition defense and improved paint coverage. The switch to drop coverage against ball screens reduced corner-three frequency by 11 percent."
- *Example 2:* "Looking at the on/off data, this lineup combination outscores opponents by 12.3 per 100 possessions over 200 tracked possessions. That sample is large enough to treat as signal. The coach should be running it in crunch time."

**`hot_take`** — The author's primary purpose is to argue or convince. They state a bold, controversial, or unpopular claim and support it. The opinion leads; any statistics serve as ammunition rather than as the point itself. No anchor to a specific live event is required.

- *Example 1:* "This player was handed two championships by the best supporting cast of the modern era. Strip that away and nobody puts him in the top ten all-time."
- *Example 2:* "The analytics revolution has made basketball worse to watch. Hunting corner threes and paint finishes for three hours is not interesting basketball."

**`reaction`** — The author's primary purpose is to express an emotional response to a specific, just-happened event — a game, a play, a trade announcement, or breaking injury news. The post is time-anchored and urgent; it could not have been written a week later without the specific triggering moment.

- *Example 1:* "That buzzer-beater tonight made me cry. I have been a fan through the bad years and this is what it is all for. I cannot stop shaking."
- *Example 2:* "Just left the arena. My voice is gone. My ears are ringing. That overtime was the best basketball I have ever watched live."

---

## Dataset

**Source:** Reddit r/nba posts and top-level comments, collected manually across game threads (reaction-rich), weekly discussion and Film Room posts (analysis-rich), and general discussion and Unpopular Opinion threads (hot_take-rich).

**Size:** 210 labeled examples

**Label distribution:**

| Label | Count |
|---|---|
| analysis | 70 |
| hot_take | 70 |
| reaction | 70 |

**Split:** 70% train / 15% validation / 15% test (handled automatically by the notebook)

**Labeling process:** Each post was read individually and labeled according to the definitions above. An LLM (Claude) was used to pre-label an initial batch; every pre-assigned label was reviewed and corrected before finalizing.

### Difficult-to-Label Examples

**1. Stat-backed argument:** "No team outside the four largest markets has won a championship since 2004. At some point that stops being a coincidence and starts being a structural feature." — This cites a real statistic but uses it as proof of a structural claim. The stat is ammunition, not explanation. **Labeled: hot_take.**

**2. Emotional reaction with opinion:** "That ejection just killed us. We were up five with eight minutes to go and now we are trying to close without our best defender. I am furious." — High emotional register and opinion about the call, but it is firmly anchored to a specific live game moment. **Labeled: reaction.**

**3. Opinion anchored to a recent event:** "This trade deadline was the worst performance by a front office I have seen in years. They sold assets at a discount and bought liabilities at a premium." — Trade deadline is an event, but the post is making a general evaluative argument with no time-urgency of reaction. **Labeled: hot_take.**

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:** Fine-tuned on Google Colab (free T4 GPU) using the HuggingFace `transformers` and `datasets` libraries with `scikit-learn` for evaluation.

**Hyperparameters:** 3 epochs, learning rate 2e-5, batch size 16. The learning rate of 2e-5 was kept at the default recommended value for DistilBERT classification fine-tuning; reducing it further did not improve validation loss given the small dataset size, and increasing it caused instability.

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt:** The prompt provided all three label definitions as written in planning.md and instructed the model to output only the label name (`analysis`, `hot_take`, or `reaction`) with no additional text.

**Baseline accuracy on test set:** 96.55%

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Zero-shot LLM baseline (llama-3.3-70b-versatile) | 96.55% |
| Fine-tuned DistilBERT | 90.62% |

The zero-shot LLM baseline outperformed the fine-tuned model by approximately 6 percentage points. This is a meaningful finding discussed in the reflection below.

### Per-Class Metrics (Fine-Tuned Model)

| Label | Precision | Recall | F1 |
|---|---|---|---|
| analysis | 0.77 | 1.00 | 0.87 |
| hot_take | 1.00 | 0.82 | 0.90 |
| reaction | 1.00 | 0.91 | 0.95 |
| **Macro avg** | **0.92** | **0.91** | **0.91** |

### Confusion Matrix (Fine-Tuned Model, Test Set)

| True \ Predicted | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | 10 | 0 | 0 |
| **hot_take** | 2 | 9 | 0 |
| **reaction** | 1 | 0 | 10 |

The dominant error pattern is hot_take posts being predicted as analysis (2 cases) and one reaction post being predicted as analysis. No posts were confused with the reaction class in the other direction.

### Wrong Predictions — Analysis

**Wrong prediction 1 (hot_take → predicted analysis):** A post arguing that teams outside large markets cannot realistically compete for championships, supported by a citation about championship winners since 2004. The model appears to have been triggered by the factual framing and the year reference — surface markers of analysis — and missed that the post's purpose was to convince rather than explain. This is the exact analysis/hot_take boundary identified in planning.md as the hardest edge case. The model learned to weight evidence-like language heavily but not the argumentative framing around it.

**Wrong prediction 2 (hot_take → predicted analysis):** A post stating that the MVP award goes to the best player on the best team every year, framed as a "hot take." The bold-claim signal ("hot take:") at the start was not enough to override the model's pattern-matching on factual-sounding language. Short posts with strong opinion framing but little structural content appear to be a systematic weakness.

**Wrong prediction 3 (reaction → predicted analysis):** A post about a coaching staff that has failed to execute a functioning play in three seasons under pressure. This post has no live-event anchor — there is no "tonight" or "just happened" signal — but the frustration is high. The model correctly identified the absence of time anchoring and predicted analysis, which is structurally more plausible than the true label reaction. On reflection, this may be a labeling edge case worth reconsidering; the post sits very close to the hot_take boundary.

### Sample Classifications

| Post (truncated) | Predicted | Confidence |
|---|---|---|
| "Their defensive rating improved 8 points since the trade deadline..." | analysis | 0.99 |
| "This player is washed. You can cite the counting stats all you want..." | hot_take | 0.97 |
| "Just left the arena. My voice is gone. My ears are ringing..." | reaction | 0.98 |
| "No team outside the five largest markets has won since 2004..." | analysis *(wrong)* | 0.81 |
| "Three overtimes and I have work at 7am and I cannot bring myself to regret..." | reaction | 0.96 |

The correctly predicted analysis post scores 0.99 confidence because it contains multiple specific statistics and no opinion framing — the model has learned that pattern cleanly. The wrong prediction scores only 0.81, showing that the model's confidence was measurably lower on the hard boundary case.

---

## Reflection: What the Model Learned vs. What Was Intended

The model learned a strong proxy for the analysis/reaction distinction that works well in practice: evidence-like language (statistics, percentages, technical terms) versus emotion-anchored language (exclamations, references to being in a physical location, direct descriptions of feeling). These proxies generalize well because they are structurally consistent across the dataset.

Where the model fell short is the analysis/hot_take boundary — exactly the hard case predicted in planning. The intended distinction is purpose: does the evidence exist to explain something, or to win an argument? The model learned instead to weight the presence of evidence-like language as a strong signal for analysis, without adequately learning the argumentative framing signals (accusatory tone, "hot take:" markers, conclusion-first structure) that distinguish a hot_take that cites a statistic from an analysis that reasons from one. The model's decision boundary captured the surface texture of evidence rather than the communicative intent behind it.

A larger dataset with more examples of the boundary cases — specifically, hot_take posts that cite statistics in service of an argument — would likely close this gap. The current dataset has approximately equal representation across all three labels but relatively few examples of the specific hot_take subtype that most closely resembles analysis.

---

## Spec Reflection

The spec's instruction to design labels before collecting any data was the single most valuable constraint in the project. Committing to precise definitions first made the annotation process significantly faster and more consistent — borderline cases could be resolved by applying the decision rules rather than relitigating the taxonomy mid-annotation.

One divergence from the spec: the spec defines success as macro F1 >= 0.88 to match or exceed the zero-shot baseline. The fine-tuned model achieves macro F1 of 0.91, which meets that threshold, but it falls below the baseline on raw accuracy (90.62% vs. 96.55%). This reversal — where the fine-tuned model has better macro F1 but lower accuracy — happens because the baseline's accuracy is partly driven by high performance on the majority of easy cases, while the fine-tuned model has better per-class balance. The success definition in planning.md focused on macro F1, so by that measure the model met the goal; but the raw accuracy comparison tells a more cautionary story about what 200 examples of fine-tuning can achieve against a state-of-the-art 70B-parameter zero-shot model.

---

## AI Usage

**Instance 1 — Label stress-testing:** Claude was given the three label definitions and the analysis/hot_take edge case description and asked to generate ten posts that sit at the boundary between those two labels. Seven of the ten generated posts could not be classified cleanly under the original definitions. This revealed that the "does the evidence explain or argue?" criterion needed an explicit tiebreaker rule. The rule added to planning.md — "if genuinely uncertain, default to hot_take" — came directly from reviewing those generated edge cases.

**Instance 2 — Pre-labeling assistance:** Claude was given the label definitions and batches of unlabeled posts and asked to assign one label per post. Approximately 60% of the pre-assigned labels were accepted without change. The remaining 40% required correction, mostly in the analysis/hot_take boundary category. The pre-labeling workflow reduced annotation time but required genuine review of every example — skimming the corrections would have produced noisy training data in exactly the region where the model most needed clean signal.

**Instance 3 — Failure pattern analysis:** After identifying the wrong predictions, Claude was given the full list of misclassified examples and asked to identify common themes. Claude suggested that short posts were overrepresented in the errors. After manually reviewing the examples, this pattern did not hold cleanly — two of the three misclassified posts were of average length. The pattern that did hold, which Claude also identified, was that all errors involved posts where evidence-like language appeared in a non-analysis context. That pattern was used as the basis for the reflection section above.

---

## Repository Contents

- `README.md` — This document (final project report)
- `planning.md` — Design notes, label definitions, edge case rules, data collection plan, evaluation metrics reasoning, and AI tool plan
- `nba_posts_labeled.csv` — Full labeled dataset (210 examples, two columns: text, label)
- `confusion_matrix.png` — Confusion matrix visualization from Colab
- `evaluation_results.json` — Side-by-side baseline vs. fine-tuned accuracy comparison
- `ai201_project3_takemeter_starter_clean.ipynb` — Training notebook (fine-tuning pipeline, baseline, evaluation)
