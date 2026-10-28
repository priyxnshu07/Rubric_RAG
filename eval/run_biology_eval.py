"""Evaluate RubricRAG on hand-labelled answers in eval/cases.jsonl.

    python -m eval.run_biology_eval --retrieval-only   # no API key needed
    python -m eval.run_biology_eval                     # full run, needs an LLM key in .env

Metrics
- Retrieval recall@k: share of the question's rubric criteria that retrieval returned.
- Score MAE / within-1: how far the LLM's total is from the teacher's reference score.
- Max-score correct: did the LLM grade exactly the right question's criteria (max = 10)?
- Citation precision: share of cited rubric criteria that belong to the question being graded.
- Rejected citations: ids the LLM invented that survived retries and were stripped.
- Human review: how many grades were flagged needs_human_review.
Results are printed and written to eval/results_biology.md.
"""
import argparse
import json
import tempfile
from pathlib import Path

from app import config, grader
from app.ingest import load_folder
from app.store import Store
from eval.run_eval import RecordingGroq

ROOT = Path(__file__).parent.parent
RUBRIC_ID = "biology-quiz-1"
RUBRIC_DIR = ROOT / "data" / RUBRIC_ID


def load_cases():
    return [json.loads(line) for line in (ROOT / "eval" / "cases.jsonl").read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--k", type=int, default=config.TOP_K_RUBRIC)
    args = ap.parse_args()
    config.TOP_K_RUBRIC = args.k

    store = Store(path=tempfile.mkdtemp(prefix="rubricrag-eval-"))
    rubric, notes, _ = load_folder(RUBRIC_DIR)
    store.ingest_documents(RUBRIC_ID, rubric, notes)
    rubric_chunks = store.get_chunks(RUBRIC_ID, "rubric")
    cases = load_cases()

    rows, recalls, feedback_samples = [], [], []
    llm = None if args.retrieval_only else RecordingGroq(config.GROQ_MODEL)  # waits out rate limits
    abs_errors, within1, max_ok, precisions, removed, reviews = [], 0, 0, [], 0, 0
    for case in cases:
        # The question's criteria: every rubric chunk whose criterion name starts with e.g. "Q1".
        expected = {c.id for c in rubric_chunks if c.criterion.startswith(case["expected_question"])}
        query = f"{case['question']}\n{case['answer']}"
        retrieved = {c.id for c in store.retrieve(RUBRIC_ID, query, "rubric", args.k, criteria_only=True)}
        recall = len(expected & retrieved) / len(expected)
        recalls.append(recall)
        row = {"id": case["id"], "recall": recall, "expected": case["expected_score"]}

        if not args.retrieval_only:
            res = grader.grade(store, RUBRIC_ID, case["question"], case["answer"], llm=llm)
            err = abs(res.score - case["expected_score"])
            abs_errors.append(err)
            within1 += err <= 1
            max_ok += res.max_score == case["expected_max"]
            cited = sorted({cid for c in res.criteria for cid in c.cited_chunk_ids if cid.startswith("R")})
            if cited:
                precisions.append(len([c for c in cited if c in expected]) / len(cited))
            removed += sum("rejected citations" in r for r in res.reasons)
            reviews += res.needs_human_review
            row.update(got=res.score, max=res.max_score,
                       review="; ".join(r if len(r) < 45 else r[:42] + "..." for r in res.reasons))
            feedback_samples.append((case["id"], res.overall_feedback))
        rows.append(row)

    n = len(cases)
    summary = [f"Cases: {n}", f"Retrieval recall@{args.k}: {sum(recalls) / n:.2f}"]
    if not args.retrieval_only:
        summary += [
            f"Score MAE: {sum(abs_errors) / n:.2f} points (out of 10)",
            f"Within 1 point of reference: {within1}/{n}",
            f"Max score correct (right criteria graded): {max_ok}/{n}",
            f"Citation precision: {sum(precisions) / len(precisions):.2f}" if precisions else "Citation precision: n/a",
            f"Criteria with invented citations rejected: {removed}",
            f"Flagged for human review: {reviews}/{n}",
            f"Model: {config.LLM_PROVIDER} / "
            f"{config.GROQ_MODEL if config.LLM_PROVIDER == 'groq' else config.GEMINI_MODEL}",
        ]

    header = "| case | recall | expected |" + ("" if args.retrieval_only else " got | max | review |")
    sep = "|---|---|---|" + ("" if args.retrieval_only else "---|---|---|")
    lines = [header, sep]
    for r in rows:
        line = f"| {r['id']} | {r['recall']:.2f} | {r['expected']} |"
        if not args.retrieval_only:
            line += f" {r['got']} | {r['max']} | {r['review']} |"
        lines.append(line)

    for cid, fb in feedback_samples[:3]:
        print(f"[{cid}] {fb}")
    report = "# Biology quiz evaluation results\n\n" + "\n".join(f"- {s}" for s in summary) + "\n\n" + "\n".join(lines) + "\n"
    print(report)
    (ROOT / "eval" / "results_biology.md").write_text(report)


if __name__ == "__main__":
    main()
