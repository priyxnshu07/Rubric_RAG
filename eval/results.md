# Grader evaluation: large vs small Groq model

Rubric: `public_speaking` (4 criteria × 2 points = 8). Answers: `eval/answers.json` (12). Model run: 2026-10-28T21:53:07+00:00. Both models use the full `/grade` logic: top-4 retrieval, schema validation with up to 2 retries, citation checks, double grading and the safeguards.

| Metric | large (`openai/gpt-oss-120b`) | small (`openai/gpt-oss-20b`) |
|---|---|---|
| Human-scored answers | 0/12 | 0/12 |
| Exact agreement with human | n/a | n/a |
| Agreement within ±1 | n/a | n/a |
| Mean absolute error (points /8) | n/a | n/a |
| Responses with only valid citations (every criterion cited) | 100% | 100% |
| Flagged for human review | 100% | 100% |
| Flagged, ignoring 'weak rubric match' | 17% | 58% |
| Failed gradings (errors) | 0 | 0 |
| Avg latency per grading (2+ LLM calls) | 3.8 s | 3.9 s |
| Avg LLM calls per grading | 2.0 | 2.1 |
| Avg tokens in / out | 2212 / 1380 | 2067 / 1927 |
| Est. cost per 100 gradings | $0.116 | $0.073 |

> **Agreement metrics are pending:** no `human_score` has been filled in yet. Fill them in `eval/answers.json`, then run `python -m eval.run_eval --from-cache`.

Cost uses list prices per 1M tokens (input/output): `openai/gpt-oss-120b` $0.15/$0.6; `openai/gpt-oss-20b` $0.075/$0.3. These are 2026 aggregator figures, not confirmed on groq.com/pricing. Every grading makes at least 2 LLM calls (double grading), more if retries were needed. Latency excludes time spent waiting out Groq free-tier rate limits.

## Per answer

| id | intended level | human | large score | small score | large flags | small flags |
|---|---|---|---|---|---|---|
| ps_01 | strong | — | 8/8 | 8/8 | weak rubric match | max_score is, not backed, weak rubric match |
| ps_02 | strong | — | 8/8 | 7/8 | weak rubric match | weak rubric match |
| ps_03 | strong | — | 8/8 | 8/8 | weak rubric match | weak rubric match |
| ps_04 | average | — | 5/8 | 6/8 | weak rubric match | not backed, weak rubric match |
| ps_05 | average | — | 7/8 | 6/8 | weak rubric match | weak rubric match |
| ps_06 | average | — | 7/8 | 7/8 | weak rubric match | weak rubric match |
| ps_07 | average | — | 5/8 | 5/8 | weak rubric match | not backed, weak rubric match |
| ps_08 | weak | — | 2/8 | 0/8 | weak rubric match | weak rubric match |
| ps_09 | weak | — | 3/8 | 3/8 | weak rubric match | not backed, weak rubric match |
| ps_10 | weak | — | 1/8 | 0/8 | weak rubric match | inconsistent scores, max_score is, not backed, weak rubric match |
| ps_11 | off-topic | — | 0/8 | 0/8 | LLM: off-topic answer, weak rubric match | not backed, weak rubric match |
| ps_12 | off-topic | — | 0/8 | 0/8 | LLM: off-topic answer, no speech script pr..., weak rubric match | LLM: Answer is off-topic and does not prov..., weak rubric match |
