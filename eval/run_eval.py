"""Evaluate RubricRAG on hand-labelled answers in eval/cases.jsonl.

    python -m eval.run_eval --retrieval-only   # no API key needed
    python -m eval.run_eval                     # full run, needs GROQ_API_KEY

Metrics
- Retrieval recall@k: share of the question's rubric lines that retrieval returned.
- Score MAE / within-1: how far the LLM's total is from the teacher's reference score.
- Max-score correct: did the LLM grade exactly the right question's criteria (max = 10)?
- Citation precision: share of cited rubric lines that belong to the question being graded.
- Removed citations: ids the LLM invented that our post-processing had to strip.
Results are printed and written to eval/results.md.
"""
import argparse
import json
import tempfile
from pathlib import Path

from app import config, grader
from app.store import Store

ROOT = Path(__file__).parent.parent
RUBRIC_ID = "biology-quiz-1"


def load_cases():
    return [json.loads(line) for line in (ROOT / "eval" / "cases.jsonl").read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--k", type=int, default=config.TOP_K_RUBRIC)
    args = ap.parse_args()
    config.TOP_K_RUBRIC = args.k

    store = Store(path=tempfile.mkdtemp(prefix="rubricrag-eval-"))
    store.ingest(RUBRIC_ID, (ROOT / "data/biology_quiz_rubric.txt").read_text(),
                 (ROOT / "data/biology_notes.txt").read_text())
    cases = load_cases()

    rows, recalls = [], []
    abs_errors, within1, max_ok, precisions, removed = [], 0, 0, [], 0
    for case in cases:
        expected = set(case["expected_lines"])
        query = f"{case['question']}\n{case['answer']}"
        retrieved = {c.id for c in store.retrieve(RUBRIC_ID, query, "rubric", args.k)}
        recall = len(expected & retrieved) / len(expected)
        recalls.append(recall)
        row = {"id": case["id"], "recall": recall, "expected": case["expected_score"]}

        if not args.retrieval_only:
            res = grader.grade(store, RUBRIC_ID, case["answer"], case["question"])
            err = abs(res.total_score - case["expected_score"])
            abs_errors.append(err)
            within1 += err <= 1
            max_ok += res.max_score == case["expected_max"]
            cited = [c.id for c in res.citations if c.id.startswith("R")]
            if cited:
                precisions.append(len([c for c in cited if c in expected]) / len(cited))
            removed += sum("removed citations" in w for w in res.warnings)
            row.update(got=res.total_score, max=res.max_score, warnings=len(res.warnings))
        rows.append(row)

    n = len(cases)
    summary = [f"Cases: {n}", f"Retrieval recall@{args.k}: {sum(recalls) / n:.2f}"]
    if not args.retrieval_only:
        summary += [
            f"Score MAE: {sum(abs_errors) / n:.2f} points (out of 10)",
            f"Within 1 point of reference: {within1}/{n}",
            f"Max score correct (right criteria graded): {max_ok}/{n}",
            f"Citation precision: {sum(precisions) / len(precisions):.2f}" if precisions else "Citation precision: n/a",
            f"Criteria with invented citations removed: {removed}",
            f"Model: {config.LLM_PROVIDER} / "
            f"{config.GROQ_MODEL if config.LLM_PROVIDER == 'groq' else config.GEMINI_MODEL}",
        ]

    header = "| case | recall | expected |" + ("" if args.retrieval_only else " got | max | warnings |")
    sep = "|---|---|---|" + ("" if args.retrieval_only else "---|---|---|")
    lines = [header, sep]
    for r in rows:
        line = f"| {r['id']} | {r['recall']:.2f} | {r['expected']} |"
        if not args.retrieval_only:
            line += f" {r['got']:g} | {r['max']:g} | {r['warnings']} |"
        lines.append(line)

    report = "# Evaluation results\n\n" + "\n".join(f"- {s}" for s in summary) + "\n\n" + "\n".join(lines) + "\n"
    print(report)
    (ROOT / "eval" / "results.md").write_text(report)


if __name__ == "__main__":
    main()
