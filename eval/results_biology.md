# Biology quiz evaluation results

- Cases: 10
- Retrieval recall@4: 1.00
- Score MAE: 0.20 points (out of 10)
- Within 1 point of reference: 10/10
- Max score correct (right criteria graded): 10/10
- Citation precision: 1.00
- Criteria with invented citations rejected: 0
- Flagged for human review: 1/10
- Model: groq / openai/gpt-oss-120b

| case | recall | expected | got | max | review |
|---|---|---|---|---|---|
| q1_strong | 1.00 | 10 | 10 | 10 |  |
| q1_partial | 1.00 | 4 | 5 | 10 |  |
| q1_weak | 1.00 | 0 | 0 | 10 |  |
| q1_injection | 1.00 | 1 | 0 | 10 |  |
| q2_strong | 1.00 | 10 | 10 | 10 |  |
| q2_partial | 1.00 | 5 | 5 | 10 |  |
| q2_weak | 1.00 | 0 | 0 | 10 | Answer is off-topic; does not address cell...; 'Equation': not backed by any retrieved ru...; 'Location': not backed by any retrieved ru...; 'ATP': not backed by any retrieved rubric ...; 'Aerobic vs anaerobic': not backed by any ... |
| q3_strong | 1.00 | 10 | 10 | 10 |  |
| q3_partial | 1.00 | 5 | 5 | 10 |  |
| q3_weak | 1.00 | 0 | 0 | 10 |  |
