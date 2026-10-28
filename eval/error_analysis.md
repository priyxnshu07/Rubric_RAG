# Error analysis: public speaking rubric, large vs small model

12 answers in `eval/answers.json`, hand-scored by me (`human_score`, 0–8), against the cached model grades in `eval/model_outputs.json`. Full table in `eval/results.md`.

| | Large `gpt-oss-120b` | Small `gpt-oss-20b` |
|---|---|---|
| Exact agreement | 50% (6/12) | 17% (2/12) |
| Within ±1 | 67% (8/12) | 58% (7/12) |
| Mean absolute error | 1.00 | 1.33 |
| Mean signed error (model − human) | **+0.83** | +0.50 |
| Over-scored / under-scored | 5 / 1 | 6 / 4 |
| MAE on "average" answers | **1.5** | 1.5 |

## The answers the models got wrong (off by 2 or more)

| Answer | Human | Large | Small | Likely cause |
|---|---|---|---|---|
| `ps_05` games club; half the speech drifts into a video game | 4 | **7** | 6 | **Missed the drift.** Both marked *Clear main message* met, but the rubric gives 1 if "the script wanders off". The ending "So that is my speech" got 0–1 correctly. |
| `ps_09` homework rant, one run-on sentence | 0 | **3** | 3 | **Generous on structure.** Both marked *Clear main message* met. The rubric gives 1 for "reasons without examples", so by its wording 1–2 points was defensible. My 0 was stricter than the rubric text. |
| `ps_03` swimming, opens with a personal story | 6 | **8** | 8 | **All four ticked.** The main message comes only after the story; I docked points that the models didn't. |
| `ps_06` dogs | 5 | **7** | 7 | **Reasons and ending given full marks.** There's one personal example (Bruno), and the call to action is tacked on. |
| `ps_08` "I like football. Football is fun…" | 2 | 2 | **0** | The small model gave nothing; the large model matched me. |

**Pattern 1: the models are lenient on middling work.** Errors concentrate on average answers (MAE 1.5) and are almost all over-scores. A criterion is "met" if the answer *mentions* the thing, even when it's done weakly.

**Pattern 2: both models fail on the same answers.** The large and small models agree with each other on `ps_03`, `ps_05`, `ps_06` and `ps_09`, and disagree with me. That points to a **rubric-reading gap** (the models read "states the main idea" generously, and I read it strictly), not random noise. For `ps_09` the rubric's own wording sides partly with the models.

## Did the safeguards catch these errors? No.

Ignoring "weak rubric match", which fired on every answer:

- **Large:** of its 4 errors of 2 or more points, **0 were flagged**. The 4 grades it flagged (`ps_08`, `ps_10`, `ps_11`, `ps_12`) were all within 1 point of my score.
- **Small:** 1 of its 5 big errors was flagged (`ps_09`), and 6 accurate grades were flagged.

So the checks catch *structural* problems (bad citations, inconsistent totals, off-topic answers), but not *judgement* errors where the model is confidently lenient. **Right now, "needs teacher review" means "something unusual", not "this score is probably wrong".** That's the most important finding of this eval.

## Other patterns

- **The small model cites course notes instead of the rubric** in 6/12 gradings. The "not backed by a rubric chunk" check caught all 6, but most of those grades were accurate, so this is a citation-quality problem rather than a scoring one.
- **The large model's totals sometimes contradict its own ticks** (`ps_08`: two criteria met, which is 4 points, but a total of 2). A check for this was added after the run. Ironically, the total of 2 matched my score; it was the ticks that were wrong.
- **Weak rubric match fired on 12/12.** Content similarity can't judge an open-topic speech against a skills rubric (see the README's Limitations).
- **Off-topic and prompt-injection answers were handled well:** 0/8 from both models, against my 1 and 0.

## What I'd do next

1. **Calibrate with examples.** Put 2–3 hand-scored example answers (one per level) in the prompt, so "met" is anchored to what the teacher means. This targets pattern 1 directly.
2. **Score each criterion 0/1/2 instead of a met/not-met tick plus a total.** This rubric is graded per criterion, and the `met` boolean loses the 1-point middle. The per-criterion scores would also make each error traceable to a specific criterion.
3. **Collect per-criterion human scores too.** I recorded only totals, so the "likely cause" column above is my reading, not measured.
4. **Fix the review flag:** flag when the two models (or two prompts) disagree by 2 or more. On this data, a large-vs-small disagreement of 2 or more would have flagged `ps_08`, but not the shared errors, so example-based calibration (1) matters more.
5. **Use more answers and a second labeller.** 12 answers and one grader (me) can't separate "the model is lenient" from "I am strict".

## Pipeline bugs this eval exposed (all fixed, with tests)

| Problem | Effect | Fix |
|---|---|---|
| The rubric's intro text took one of the top-4 retrieval slots | Every grading missed a criterion and was out of 6 | Retrieval ranks **criteria only** |
| Groq JSON mode rejected malformed output with a 400 | The grading failed instead of retrying | `json_validate_failed` goes through the normal retry loop |
| Free-tier rate limit (8k tokens/min) | Failed gradings; latency inflated to 12–20 s | The eval waits out 429s and leaves the waiting out of latency |
| Score lower than the criteria marked met | Contradiction went unnoticed | New review check |

## Verdict

**The large model is the better grader:** 50% exact against 17%, MAE 1.00 against 1.33, and it cites the rubric correctly. It costs about $0.04 more per 100 gradings. But neither model is ready to grade without a teacher: both over-score average work, and the current safeguards don't catch that.
