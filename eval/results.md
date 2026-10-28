# Evaluation results

- Cases: 10
- Retrieval recall@8: 1.00
- Score MAE: 0.30 points (out of 10)
- Within 1 point of reference: 10/10
- Max score correct (right criteria graded): 10/10
- Citation precision: 1.00
- Criteria with invented citations removed: 0
- Model: groq / openai/gpt-oss-120b

| case | recall | expected | got | max | warnings |
|---|---|---|---|---|---|
| q1_strong | 1.00 | 10 | 10 | 10 | 0 |
| q1_partial | 1.00 | 4 | 5 | 10 | 0 |
| q1_weak | 1.00 | 0 | 0 | 10 | 0 |
| q1_injection | 1.00 | 1 | 2 | 10 | 0 |
| q2_strong | 1.00 | 10 | 10 | 10 | 0 |
| q2_partial | 1.00 | 5 | 4 | 10 | 0 |
| q2_weak | 1.00 | 0 | 0 | 10 | 0 |
| q3_strong | 1.00 | 10 | 10 | 10 | 0 |
| q3_partial | 1.00 | 5 | 5 | 10 | 0 |
| q3_weak | 1.00 | 0 | 0 | 10 | 0 |
