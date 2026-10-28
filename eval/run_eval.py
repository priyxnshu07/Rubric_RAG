"""Compare a large and a small Groq model as graders on eval/answers.json.

    python -m eval.run_eval              # grade every answer with both models (calls Groq), then report
    python -m eval.run_eval --from-cache # re-report from eval/model_outputs.json, no API calls
                                         # (use this after filling in human_score)

Each answer goes through the same grader.grade() logic as POST /grade (top-4 retrieval,
schema validation with retries, citation checks, double grading, safeguards). Only the LLM
function is swapped for one that also records token usage.

Per model it reports: exact agreement and agreement within 1 point with human_score, mean
absolute error, % of LLM responses that cited only retrieved chunks, % flagged for human
review, average latency per grading, and estimated cost per 100 gradings.
Writes eval/results.md and caches raw results in eval/model_outputs.json.
"""
import argparse
import json
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app import config, grader
from app.ingest import load_folder
from app.schemas import GradeResponse
from app.store import Store

ROOT = Path(__file__).parent.parent
ANSWERS = ROOT / "eval" / "answers.json"
CACHE = ROOT / "eval" / "model_outputs.json"
RESULTS = ROOT / "eval" / "results.md"


@dataclass
class ModelSpec:
    name: str
    usd_per_m_input: float
    usd_per_m_output: float


# Groq list prices per 1M tokens (2026 aggregator figures, e.g. morphllm.com/groq-pricing, October 2026).
# Groq's own launch blog listed higher prices; check groq.com/pricing before relying on the cost numbers.
MODELS = {
    "large": ModelSpec("openai/gpt-oss-120b", 0.15, 0.60),
    "small": ModelSpec("openai/gpt-oss-20b", 0.075, 0.30),
}


class RecordingGroq:
    """An LLM function for grader.grade() that also records token usage and raw replies.

    Free-tier Groq limits tokens per minute, so on a 429 it waits as long as Groq asks and
    tries again. Time spent waiting is tracked so it can be left out of the latency figures.
    """

    def __init__(self, model: str):
        from groq import Groq

        if not config.GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY is not set in .env")
        # max_retries=0: the SDK's own silent back-off would otherwise be counted as latency.
        self.client, self.model, self.calls = Groq(api_key=config.GROQ_API_KEY, max_retries=0), model, []
        self.waited_s = 0.0
        self.rate_limit_waits = 0

    def __call__(self, messages: list[dict]) -> str:
        from groq import RateLimitError

        for _ in range(8):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0, response_format={"type": "json_object"})
                break
            except RateLimitError as e:
                m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
                wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else 20
                self.rate_limit_waits += 1
                self.waited_s += min(wait, 90)
                time.sleep(min(wait, 90))
            except Exception as e:
                failed = grader.groq_failed_generation(e)
                if failed is not None:  # same handling as grader.groq_llm(): retried by call_llm()
                    self.calls.append({"input_tokens": 0, "output_tokens": 0, "raw": failed})
                    return failed
                raise grader.LLMProviderError(f"Groq API error: {e}") from e
        else:
            raise grader.LLMProviderError("Groq API error: still rate limited after 8 waits")
        text = resp.choices[0].message.content or ""
        self.calls.append({"input_tokens": resp.usage.prompt_tokens,
                           "output_tokens": resp.usage.completion_tokens, "raw": text})
        return text


def citation_check(raw_replies: list[str], allowed: set[str]) -> tuple[int, int]:
    """(parsed responses, responses where EVERY criterion cites at least one id and ALL ids were retrieved).

    Measured on raw LLM replies, before our code strips anything. A criterion citing nothing counts as invalid.
    """
    parsed = valid = 0
    for raw in raw_replies:
        try:
            res = GradeResponse.model_validate_json(grader._strip_code_fence(raw))
        except ValidationError:
            continue
        parsed += 1
        valid += all(c.cited_chunk_ids and all(cid in allowed for cid in c.cited_chunk_ids) for c in res.criteria)
    return parsed, valid


def run_models(answers: list[dict]) -> dict:
    store = Store(path=tempfile.mkdtemp(prefix="rubricrag-eval-"))
    for coll in sorted({a["rubric_collection"] for a in answers}):
        rubric, notes, _ = load_folder(ROOT / "data" / coll)
        store.ingest_documents(coll, rubric, notes)

    out = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}
    for key, spec in MODELS.items():
        llm = RecordingGroq(spec.name)
        rows = []
        for a in answers:
            q, ans, coll = a["question"], a["answer"], a["rubric_collection"]
            # Same retrieval as grader.grade(), to know which chunk ids were allowed.
            allowed = {c.id for c in store.retrieve(coll, f"{q}\n{ans}", "rubric", config.TOP_K_RUBRIC,
                                                    criteria_only=True)
                       + store.retrieve(coll, f"{q}\n{ans}", "notes", config.TOP_K_NOTES)}
            llm.calls, llm.waited_s = [], 0.0
            started = time.perf_counter()
            row = {"id": a["id"]}
            try:
                res = grader.grade(store, coll, q, ans, llm=llm)
                row.update(score=res.score, max_score=res.max_score, needs_review=res.needs_human_review,
                           reasons=res.reasons, overall_feedback=res.overall_feedback,
                           criteria=[c.model_dump() for c in res.criteria])
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            row["latency_s"] = round(time.perf_counter() - started - llm.waited_s, 2)  # excludes rate-limit waits
            row["rate_limit_wait_s"] = round(llm.waited_s, 1)
            row["llm_calls"] = len(llm.calls)
            row["input_tokens"] = sum(c["input_tokens"] for c in llm.calls)
            row["output_tokens"] = sum(c["output_tokens"] for c in llm.calls)
            row["responses_parsed"], row["responses_valid_citations"] = citation_check(
                [c["raw"] for c in llm.calls], allowed)
            rows.append(row)
            print(f"{key:<5} {a['id']}: score={row.get('score', 'ERR')} review={row.get('needs_review')} "
                  f"{row['latency_s']}s")
        out["models"][key] = {"model": spec.name, "usd_per_m_input": spec.usd_per_m_input,
                              "usd_per_m_output": spec.usd_per_m_output, "results": rows}
    return out


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def summarise(m: dict, human: dict[str, int | None]) -> dict:
    rows = m["results"]
    ok = [r for r in rows if "error" not in r]
    scored = [r for r in ok if human.get(r["id"]) is not None]
    diffs = [abs(r["score"] - human[r["id"]]) for r in scored]
    parsed = sum(r["responses_parsed"] for r in rows)
    cost_each = [r["input_tokens"] * m["usd_per_m_input"] / 1e6 + r["output_tokens"] * m["usd_per_m_output"] / 1e6
                 for r in rows]
    return {
        "Human-scored answers": f"{len(scored)}/{len(rows)}",
        "Exact agreement with human": pct(sum(d == 0 for d in diffs), len(scored)),
        "Agreement within ±1": pct(sum(d <= 1 for d in diffs), len(scored)),
        "Mean absolute error (points /8)": f"{sum(diffs) / len(diffs):.2f}" if diffs else "n/a",
        "Responses with only valid citations (every criterion cited)": pct(sum(r["responses_valid_citations"] for r in rows), parsed),
        "Flagged for human review": pct(sum(r["needs_review"] for r in ok), len(ok)),
        "Flagged, ignoring 'weak rubric match'": pct(
            sum(any(x != "weak rubric match" for x in r["reasons"]) for r in ok), len(ok)),
        "Failed gradings (errors)": str(len(rows) - len(ok)),
        "Avg latency per grading (2+ LLM calls)": f"{sum(r['latency_s'] for r in rows) / len(rows):.1f} s",
        "Avg LLM calls per grading": f"{sum(r['llm_calls'] for r in rows) / len(rows):.1f}",
        "Avg tokens in / out": (f"{sum(r['input_tokens'] for r in rows) / len(rows):.0f} / "
                                f"{sum(r['output_tokens'] for r in rows) / len(rows):.0f}"),
        "Est. cost per 100 gradings": f"${100 * sum(cost_each) / len(rows):.3f}",
    }


def short_reason(r: str) -> str:
    for key in ["inconsistent scores", "weak rubric match", "not retrieved", "rejected citations",
                "not backed", "max_score is"]:
        if key in r:
            return key
    return "LLM: " + (r if len(r) <= 40 else r[:37] + "...")


def report(cache: dict, answers: list[dict]) -> str:
    human = {a["id"]: a.get("human_score") for a in answers}
    models = cache["models"]
    summaries = {k: summarise(m, human) for k, m in models.items()}
    head = " | ".join(f"{k} (`{m['model']}`)" for k, m in models.items())
    lines = [
        "# Grader evaluation: large vs small Groq model",
        "",
        f"Rubric: `public_speaking` (4 criteria × 2 points = 8). Answers: `eval/answers.json` "
        f"({len(answers)}). Model run: {cache['run_at']}. Both models use the full `/grade` logic: "
        f"top-{config.TOP_K_RUBRIC} retrieval, schema validation with up to {grader.MAX_RETRIES} retries, citation "
        f"checks, double grading and the safeguards.",
        "",
        f"| Metric | {head} |",
        "|---|" + "---|" * len(models),
    ]
    for metric in next(iter(summaries.values())):
        lines.append(f"| {metric} | " + " | ".join(s[metric] for s in summaries.values()) + " |")
    if not any(v is not None for v in human.values()):
        lines += ["", "> **Agreement metrics are pending:** no `human_score` has been filled in yet. "
                      "Fill them in `eval/answers.json`, then run `python -m eval.run_eval --from-cache`."]
    lines += [
        "",
        "Cost uses list prices per 1M tokens (input/output): " + "; ".join(
            f"`{m['model']}` ${m['usd_per_m_input']}/${m['usd_per_m_output']}" for m in models.values())
        + ". These are 2026 aggregator figures, not confirmed on groq.com/pricing. Every grading makes at "
          "least 2 LLM calls (double grading), more if retries were needed. Latency excludes time spent "
          "waiting out Groq free-tier rate limits.",
        "",
        "## Per answer",
        "",
        "| id | intended level | human | " + " | ".join(f"{k} score" for k in models) + " | "
        + " | ".join(f"{k} flags" for k in models) + " |",
        "|---|---|---|" + "---|" * (2 * len(models)),
    ]
    by_model = {k: {r["id"]: r for r in m["results"]} for k, m in models.items()}
    for a in answers:
        cells = []
        for k in models:
            r = by_model[k][a["id"]]
            cells.append("error" if "error" in r else f"{r['score']}/{r['max_score']}")
        flags = []
        for k in models:
            r = by_model[k][a["id"]]
            flags.append("" if "error" in r else ", ".join(sorted({short_reason(x) for x in r["reasons"]})))
        hs = a.get("human_score")
        lines.append(f"| {a['id']} | {a.get('intended_level', '')} | {'—' if hs is None else hs} | "
                     + " | ".join(cells) + " | " + " | ".join(flags) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true", help="re-report from eval/model_outputs.json")
    args = ap.parse_args()
    answers = json.loads(ANSWERS.read_text())

    if args.from_cache:
        cache = json.loads(CACHE.read_text())
    else:
        cache = run_models(answers)
        CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    md = report(cache, answers)
    RESULTS.write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
