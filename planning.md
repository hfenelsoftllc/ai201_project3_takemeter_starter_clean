# planning.md — TakeMeter Project (AI201 Project 3)

## 1. Community

**Chosen community: r/nba (Reddit)**

r/nba is one of the largest sports communities on the internet, with over 9 million members generating thousands of posts and comments on any given game day. It was chosen for this project because NBA basketball inspires three genuinely distinct modes of written expression that appear in roughly equal proportion: rigorous analytical discourse citing statistics and advanced metrics, opinionated argumentation about players and franchises, and real-time emotional reactions to live events. This variety makes the community a strong fit for a classification task — a model that can distinguish these modes could power tools like automatic post tagging, thread routing (send reaction posts to game threads, analysis to weekly discussion threads), or community-level discourse health metrics. The discourse is interesting precisely because the three modes are superficially similar — all use basketball vocabulary, all appear on the same platform — but differ sharply in purpose, structure, and the social response they invite, making surface-level keyword matching insufficient and requiring genuine semantic understanding.

---

## 2. Labels

Three mutually exclusive labels, distinguished by the **author's primary purpose**:

### Label: analysis
**Definition:** The author's primary purpose is to *explain* something about basketball using evidence — statistics, advanced metrics, film observations, lineup data, or tactical breakdowns. The evidence leads; any opinion is incidental.

**Example 1:**
> Their defensive rating improved 8 points since the trade deadline, driven primarily by better transition defense and improved paint coverage. The switch to drop coverage against ball screens reduced corner-three frequency by 11 percent.

**Example 2:**
> Looking at the on/off data, this lineup combination outscores opponents by 12.3 per 100 possessions over 200 tracked possessions. That sample is large enough to treat as signal. The coach should be running it in crunch time.

---

### Label: hot_take
**Definition:** The author's primary purpose is to *argue* or *convince* — they state a bold, controversial, or unpopular claim and support it. The opinion leads; any statistics serve as ammunition rather than as the point itself. No anchor to a specific live event is required.

**Example 1:**
> This player was handed two championships by the best supporting cast of the modern era. Strip that away and nobody puts him in the top ten all-time. The era flatters him and historians will eventually correct the record.

**Example 2:**
> The analytics revolution has made basketball worse to watch. Hunting corner threes and paint finishes for three hours is not interesting basketball. Efficiency optimisation and entertainment are not the same thing.

---

### Label: reaction
**Definition:** The author's primary purpose is to *express an emotional response* to a specific, just-happened event — a game, a play, a trade announcement, breaking injury news. The post is time-anchored and urgent; it could not have been written a week later without the specific triggering moment.

**Example 1:**
> That buzzer-beater tonight made me cry. I have been a fan through the bad years and this is what it is all for. I cannot stop shaking. I do not even care what happens the rest of the season.

**Example 2:**
> Did anyone else just completely lose it on that alley-oop? I jumped off my couch and scared my dog. I have watched it twenty times since the game ended and it gets better every view.

---

## 3. Hard Edge Cases

### Primary ambiguity: analysis vs. hot_take

The hardest edge case is a post that **cites statistics in service of an argument**. Both analysis and hot_take posts can reference advanced metrics, historical comparisons, and win probability figures. The distinguishing question is: does the evidence exist to explain a phenomenon, or to win an argument?

A post like *Teams outside the five largest markets have not won a championship since 2004 — at some point that stops being a coincidence and starts being a structural feature* cites a real statistic but frames it as proof of a structural claim. The stat is ammunition. That is a hot_take. By contrast, *Their win probability collapses in fourth quarters because their clutch net rating is minus-3.1 — here is why that happens mechanically* uses evidence to explain a phenomenon. That is analysis.

**Annotation decision rule:** Ask: is the author trying to make me *agree with a position*, or trying to make me *understand something*? If agreement is the goal, label it hot_take. If understanding is the goal, label it analysis. When genuinely uncertain after applying this rule, default to hot_take, since misclassifying a hot_take as analysis is the more harmful error for downstream use — it would cause a community tool to surface argumentative posts in analytical discussion threads.

### Secondary ambiguity: hot_take vs. reaction

A post can be emotionally charged and opinionated without being tied to a live event. *This team's fans are the most delusional in sports* is emotional but is a general ongoing opinion — label it hot_take. The test for reaction is strict: **there must be a specific triggering event the post cannot exist without**. Look for time anchors: tonight, just saw, that play, the trade just dropped, I was there. If none are present, the post defaults to hot_take even if the emotional register is high.

---

## 4. Data Collection Plan

### Sources

Primary source: Reddit r/nba posts and top-level comments, collected manually by browsing the subreddit across several session types:
- **Game threads** (during and immediately after games) — rich source of reaction posts
- **Weekly discussion threads** and Film Room posts — rich source of analysis posts
- **General discussion posts** and Unpopular Opinion threads — rich source of hot_take posts

Secondary source: Historical posts via Reddit search using query terms such as *hot take:*, *unpopular opinion:*, *explain why*, *advanced metrics*, to fill underrepresented labels.

### Target quantities

- 70 examples per label x 3 labels = **210 total examples** (minimum; target 75 per label for annotation headroom)
- All examples must be at least one complete sentence; no single-word or emoji-only posts
- All examples must be original Reddit text, not paraphrased or synthetic

### Handling underrepresentation

If after 200 examples any label has fewer than 60 examples, apply targeted collection:
- For analysis: search r/nba for posts containing *per 100 possessions*, *net rating*, *advanced metrics*, *TS%*, *lineup data*
- For hot_take: search for *unpopular opinion*, *hot take*, *controversial:* threads and weekly debate posts
- For reaction: collect from game thread archives on days of high-profile games or major trades

Do not synthetically generate examples or paraphrase — all examples must reflect authentic community voice to avoid training on text that does not match the real distribution.

---

## 5. Evaluation Metrics

### Primary metric: Macro F1-score

Accuracy alone is insufficient because errors on each class have different costs and the real-world label distribution may differ from the training set. Macro F1 weights each class equally regardless of support, making it the right primary metric for a task where all three labels are equally important to classify correctly. A model that achieves 90% accuracy by predicting *analysis* on a skewed dataset is not useful — macro F1 penalises that collapse.

### Per-class precision and recall

For a community tool, the *direction* of error matters:
- **False positives on hot_take** (mislabelling analysis as hot_take) are costly: analytical posts get routed to debate threads where they receive bad-faith replies instead of substantive engagement.
- **False negatives on reaction** (missing a reaction and labelling it hot_take) are less costly: the post still reaches a general discussion context where it fits reasonably well.

Reporting precision and recall per class allows the classification threshold to be tuned for the specific deployment context and makes asymmetric error costs visible.

### Confusion matrix

Required to identify which pairs of classes are being confused. The analysis/hot_take pair is the expected failure mode given label proximity; the confusion matrix makes this pattern visible and directly informs further dataset cleaning or prompt improvements.

### Zero-shot LLM baseline comparison

The fine-tuned model must be compared against the Groq zero-shot baseline. If the fine-tuned model underperforms the LLM on macro F1, that is a meaningful finding: it tells us whether fine-tuning on this dataset adds value over what a general-purpose LLM already understands from pretraining.

---

## 6. Definition of Success

### Minimum threshold for deployment

A classifier is deployable as a community tool if it achieves:
- **Macro F1 >= 0.80** across all three labels on the held-out test set
- **Per-class F1 >= 0.75** for every individual label (no class below this floor)
- **Hot_take precision >= 0.80** (control false-positives that route analytical posts into debate contexts)

At this threshold, roughly 1 in 5 posts would be misclassified at worst, which is acceptable for a soft tool — one that makes suggestions and allows user override — rather than an enforcement mechanism.

### Genuinely useful bar

For the classifier to be genuinely valuable — not just technically functional — it should achieve **macro F1 >= 0.88**, matching or exceeding the zero-shot LLM baseline on the held-out set. This level of performance means the fine-tuned model has learned community-specific language patterns beyond what a general-purpose LLM already knows from pretraining, which justifies the annotation and fine-tuning investment over simply calling the Groq API at inference time.

### What failure looks like

A classifier that conflates analysis and hot_take at high rates — confusion matrix showing more than 20% of one class predicted as the other — is not deployable regardless of overall accuracy, because that is precisely the distinction the tool exists to make. High accuracy achieved by collapsing two classes is the specific failure mode to guard against in this task.