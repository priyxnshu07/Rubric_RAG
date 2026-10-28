# RubricRAG

A rubric-grounded feedback assistant. A teacher uploads a rubric and course notes; a student answer is graded by an LLM using retrieved rubric chunks, returning a score, per-criterion feedback, and citations to the rubric lines used.

## Problem

Asking an LLM "grade this answer" gives scores that are hard to trust: the model invents its own criteria, its arithmetic can be wrong, and nothing shows *why* a student lost marks. Teachers already have the answer to this problem, which is a rubric. RubricRAG makes the LLM grade **only against the teacher's rubric**, and every criterion score points back to the exact rubric line (and line number) it was based on, so a teacher can check it in seconds and a student knows exactly what to improve.

## Architecture

```
          Teacher                                   Student answer
             │ rubric.txt + notes.txt                     │
             ▼                                            ▼
   ┌──────────────────┐                        ┌──────────────────────┐
   │ Streamlit UI     │ ──── HTTP (JSON) ────▶ │ FastAPI  (app/main)  │
   │ (app/ui.py)      │ ◀─────────────────────  │ /ingest  /grade      │
   └──────────────────┘                        └──────────┬───────────┘
                                                           │
             ┌─────────────────────────────────────────────┼───────────────────────┐
             ▼                                             ▼                       ▼
   ┌──────────────────────┐   embed (MiniLM)   ┌──────────────────────┐   ┌─────────────────┐
   │ Ingest (.md/.txt/pdf)│ ─────────────────▶ │ ChromaDB (local)     │   │ Groq / Gemini   │
   │ rubric: 1/criterion  │                    │ filtered by          │   │ JSON output,    │
   │ notes: 1 per para    │                    │ rubric_id + doc_type │   │ temperature 0   │
   └──────────────────────┘                    └──────────────────────┘   └────────┬────────┘
                                                                                    ▼
                                                                  ┌─────────────────────────────┐
                                                                  │ Post-processing (in code)   │
                                                                  │ • drop invented citations   │
                                                                  │ • clamp scores to max       │
                                                                  │ • sum total in Python       │
                                                                  └─────────────────────────────┘
```

| Layer | Tool | File |
|---|---|---|
| API | FastAPI | `app/main.py` |
| Ingestion (CLI + chunking) | pypdf for PDFs | `app/ingest.py`, `app/chunking.py`, `app/loaders.py` |
| Vector store | ChromaDB (local, persistent) | `app/store.py` |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | `app/store.py` |
| LLM | Groq or Google Gemini, chosen by `LLM_PROVIDER` | `app/grader.py` |
| Schemas | Pydantic | `app/schemas.py` |
| UI | Streamlit | `app/ui.py` |
| Tests | pytest | `tests/` |

```
app/     API, ingestion CLI, chunking, vector store, grader, UI
data/    one folder per rubric (rubric + notes), plus example answers
         biology-quiz-1/rubric.md, notes.md
eval/    labelled test cases + evaluation script
tests/   pytest unit and integration tests (no API key needed)
```

## How it works

1. **Ingest.** `python -m app.ingest data/<folder>` (or upload in the UI) loads the rubric and notes from `.md`, `.txt` or `.pdf`. The rubric is split **one chunk per criterion, never across criteria** (see *Chunking choice* below). Notes are split into paragraphs. Each chunk keeps its line range and an id like `R7-11` (rubric lines 7–11) or `N4-6` (notes lines 4–6).
2. **Embed and store.** Chunks are embedded with `all-MiniLM-L6-v2` and stored in ChromaDB with metadata: `chunk_id`, `source` file, `criterion` name, `points`, `doc_type`, `rubric_id` and line numbers. Re-ingesting a rubric replaces its old chunks.
3. **Retrieve.** The question and answer text form the query. The **top-4 rubric criteria** and top-3 notes paragraphs for that `rubric_collection` are retrieved. One rubric can cover a whole quiz, so retrieval is what picks out the right question's criteria.
4. **Grade.** All instructions go in the **system prompt**. The **user message** holds only data: the chunks tagged with their ids, the question, and the answer wrapped in `<answer>` tags. That way student text never mixes with the instructions. The LLM (Groq by default, or Gemini; the grader takes any function `messages -> text`) returns JSON in the response schema below.
5. **Validate and retry.** The JSON is validated against the Pydantic schema, including a rule that `score` can't exceed `max_score`. If it's invalid, or cites a chunk id that wasn't retrieved, the bad output and the exact error are sent back to the model ("`criteria.0.met`: Field required", or "`R99` is not in the context, only cite: …"). It gets **up to 2 retries**; after that, the request fails with a clear error.
6. **Verify in code.** Checks the LLM can't be trusted with. If any fails, `needs_human_review` is set and the reason is listed:
   - a citation that survived the retries isn't in the retrieved set (it's rejected);
   - a criterion isn't backed by any retrieved rubric chunk;
   - `max_score` doesn't equal the points of the cited criteria;
   - **some of the question's criteria weren't retrieved**, so the answer was graded out of fewer points than the question is worth.

   The LLM can also flag a grade for review itself, for example an off-topic answer or one trying to instruct the grader.

### Safeguards

| # | Safeguard | How | Config (`.env`) |
|---|---|---|---|
| 1 | **Consistency** | Every answer is graded **twice**; if the two scores differ by more than 1, it's flagged `"inconsistent scores"`. The first grade is returned. | `MAX_SCORE_DISAGREEMENT=1` |
| 2 | **Weak rubric match** | The **answer alone** is embedded and compared with the rubric. If even its closest criterion has cosine similarity below the threshold, it's flagged `"weak rubric match"`. | `MIN_RUBRIC_SIMILARITY=0.35` |
| 3 | **Age-appropriate feedback** | The system prompt says students are 8–14: start with what went well, use simple words, never be harsh, and don't let encouragement change the score. | |
| 4 | **Privacy-safe logging** | Logs hold only a request id, collection name, chunk ids, both scores, similarity and latency, never the question, the answer or the LLM's reasons (they may quote the answer). Failures log only the error type. | |

**Why the answer alone for #2:** I measured the best similarity on the sample data. With question + answer as the query, "I don't know." scored **0.63**, as high as real answers (0.50–0.68), because the question always matches its own rubric. With the answer alone, real answers scored **0.46–0.68** and off-topic ones **≤ 0.25**, so 0.35 sits in the gap. Retrieval still uses question + answer, because that's better for picking the right question's criteria.

Example log line:
```
INFO rubricrag.grade: grade ok request_id=3f9c1a2b collection=biology-quiz-1 retrieved=['N14-15', 'R30-34', ...] cited=['R36-38'] scores=2,2 max_score=10 best_similarity=0.61 needs_review=False latency_ms=2140
```

### API

`POST /grade`

```json
{"question": "Where does cellular respiration happen?", "answer": "In the mitochondria.", "rubric_collection": "biology-quiz-1"}
```

```json
{
  "score": 2, "max_score": 10,
  "criteria": [{"name": "Location", "met": true, "feedback": "Correct.", "cited_chunk_ids": ["R36-38"]}],
  "overall_feedback": "...",
  "needs_human_review": false,
  "reasons": []
}
```

| Status | When |
|---|---|
| 200 | Graded (check `needs_human_review`) |
| 404 | `rubric_collection` hasn't been ingested |
| 422 | Request is missing `question`, `answer` or `rubric_collection` |
| 500 | Server misconfigured, e.g. API key missing |
| 502 | LLM provider failed, or returned invalid JSON 3 times (the message includes the last validation error) |

`GET /rubrics/{rubric_collection}/chunks` returns the chunk text behind each `cited_chunk_id`.

### Chunking choice

A criterion is the unit a teacher grades by, so it's also the unit of retrieval and citation:

- **A criterion starts at any line that states its points**, such as `### Inputs (3 points)` or `Q1 Inputs (3 pts): ...`. It runs until the next criterion or heading, so its level descriptors stay with it, even across blank lines. Splitting by a fixed token size could separate "Inputs (3 points)" from "2: names two of these", and then the LLM would see a score level with no criterion.
- **Headings without points are sections** (e.g. `## Q1 Photosynthesis`). They aren't chunks themselves; they're added as a prefix to each criterion under them. That gives the embedding the question context, so an answer about photosynthesis retrieves Q1's criteria, not Q2's.
- **Any other text** (title, instructions) becomes a "general" chunk with no criterion name.
- **Trade-off:** this relies on the rubric stating points in brackets. A rubric without them would come out as general paragraphs: still searchable, but without criterion names.

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add a GEMINI_API_KEY or GROQ_API_KEY and set LLM_PROVIDER
```

### Run

If a model name stops working (providers rename models), set `GEMINI_MODEL` / `GROQ_MODEL` in `.env` to a current one.

```bash
python -m app.ingest data/biology-quiz-1   # load the sample rubric + notes into ChromaDB
uvicorn app.main:app --reload          # API at http://localhost:8000 (docs at /docs)
streamlit run app/ui.py                # UI at http://localhost:8501 (second terminal)
```

`biology-quiz-1` then appears in the UI's Grade tab; paste an answer from `data/sample_answers.md`. You can also upload a rubric and notes in the UI's Teacher tab.

A rubric folder needs exactly one file with `rubric` in its name and any number with `notes` in their name. Other files are skipped, and the folder name becomes the rubric id.

### Test

```bash
pytest
```

The tests use a fake LLM, so they need no API key. They do use the real embedding model and a temporary ChromaDB.

## Evaluation

`eval/cases.jsonl` holds 10 student answers across the 3 quiz questions (strong, partial and weak for each, plus one prompt-injection attempt). I labelled each with a reference score by applying the rubric by hand.

```bash
python -m eval.run_eval --retrieval-only   # retrieval metrics only, no API key
python -m eval.run_eval                    # full pipeline, needs an LLM key
```

| Metric | What it checks |
|---|---|
| Retrieval recall@k | Did retrieval return all of the question's rubric criteria? |
| Score MAE / within 1 point | How close the LLM's total is to my reference score |
| Max score correct | Did the LLM grade the right question's criteria (max = 10)? |
| Citation precision | Share of cited rubric criteria that belong to the right question |
| Invented citations rejected | Fake citations that survived the retries |
| Flagged for review | Grades marked `needs_human_review` |

**Results:** see `eval/results.md` (written by the script).

- **Retrieval:** recall@k = **0.97** at k = 4 (the endpoint's setting) and **1.00** at k = 6 and 8. With only 4 slots, one answer (`q2_partial`) missed one of its question's 4 criteria.
- **Grading (LLM):** full pipeline with Groq `openai/gpt-oss-120b`:

| Metric | Result |
|---|---|
| Score MAE (vs my reference scores) | **0.50** points out of 10 |
| Within 1 point of reference | **9/10** |
| Right criteria graded (max score = 10) | **9/10** |
| Citation precision | **1.00** |
| Invented citations rejected | 0 |
| Flagged for human review | 3/10, each for a real reason (below) |
| Prompt-injection answer ("award full marks") | scored **2/10** (reference 1), flagged "inconsistent scores" |
| "Weak rubric match" on genuine answers | 0/10 (no false alarms); an off-topic answer is flagged in `tests/` |

The one real miss is `q2_partial`. Top-4 retrieval dropped one of Q2's four criteria, so it was graded out of 8 and scored 2 against my reference of 5. Before I added the "criteria not retrieved" check, this failed **silently**: the score looked internally consistent. Now it's flagged for human review. All strong and weak answers were scored exactly. With k = 8 (an earlier version), the same set reached MAE 0.30 and 10/10 within 1 point, so top-4 trades a little accuracy for a shorter prompt. The other two flags were the prompt-injection answer (its two runs disagreed, which suggests the attack makes grading unstable) and an off-topic answer the LLM flagged itself. The exact numbers move slightly between runs (MAE was 0.40 on the previous run), which is exactly why safeguard #1 exists. This was 10 cases, so treat it as a sanity check rather than a benchmark.

## Limitations

- **Small eval set, one labeller.** 10 cases labelled by me is a sanity check, not proof. A real evaluation needs many answers marked by several teachers, with agreement between them measured.
- **Top-4 retrieval can miss a criterion.** It's flagged for review, not fixed. A real fix would be to retrieve the whole question section once any of its criteria matches, using the section metadata already stored.
- **The LLM still judges.** The code catches invented citations, inconsistent max scores and missing criteria, but it can't catch a wrong judgement on a valid criterion. Teachers should review the scores, not rely on them blindly.
- **Document formats.** `.md`, `.txt` and text-based `.pdf` only. Scanned PDFs (images) and `.docx` aren't supported, and PDF line numbers refer to the extracted text.
- **Criterion detection relies on "(N points)".** Rubrics that don't state points per criterion fall back to paragraph chunks without criterion names.
- **No auth, single user.** Anyone who can reach the API can upload or grade. Fine for a local demo, not for deployment.
- **The consistency check is weak at temperature 0.** Both runs use the same prompt and temperature 0, so they usually agree; the check only catches the model's own randomness. It also doubles cost and latency. A stronger version would vary the prompt or the order of the chunks between runs.
- **The similarity threshold was tuned on tiny data.** 0.35 comes from 10 answers and 4 off-topic examples from one subject. It should be re-tuned per subject.
- **Age-appropriate tone is prompted, not checked.** Nothing measures reading level. A readability score (e.g. Flesch–Kincaid) on the feedback would be a cheap next check.
- **Prompt injection is mitigated, not solved.** The answer is clearly delimited and the model is told to ignore instructions inside it, but a determined attack could still sway a score.
