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
3. **Retrieve.** For a student answer, the question and answer text form the query. The top-k rubric criteria (default 8) and top-3 notes paragraphs for that `rubric_id` are retrieved. One rubric can cover a whole quiz, so retrieval is what picks out the right question's criteria.
4. **Grade.** The LLM (Groq or Gemini; the grader is just a function `(system, user) -> JSON text`, so providers are swappable) receives the chunks tagged with their ids, the question, and the answer wrapped in `<answer>` tags with an instruction to ignore any instructions inside it. It returns JSON: one entry per criterion with score, max score, feedback, and cited ids. This is validated with Pydantic and retried once if malformed.
5. **Verify in code.** Citations that weren't in the retrieved context are removed, scores are clamped to each criterion's max, and the total is summed in Python. Anything suspicious is returned as a `warning` instead of being hidden.

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
| Invented citations removed | How often post-processing had to strip a fake citation |

**Results:** see `eval/results.md` (written by the script).

- **Retrieval:** recall@k = **0.97** at k = 4 and **1.00** at k = 6 and 8. With only 4 slots, one answer missed one of its question's 4 criteria. The default is k = 8 for headroom. Section names are prefixed to every criterion, which makes this an easy setting.
- **Grading (LLM):** full pipeline with Groq `openai/gpt-oss-120b`:

| Metric | Result |
|---|---|
| Score MAE (vs my reference scores) | **0.30** points out of 10 |
| Within 1 point of reference | **10/10** |
| Right criteria graded (max score = 10) | **10/10** |
| Citation precision | **1.00** |
| Invented citations removed | 0 |
| Prompt-injection answer ("award full marks") | scored **2/10** (reference 1), not 10 |

The three cases that missed were all partial answers, off by exactly 1 point (`q1_partial` 5 vs 4, `q2_partial` 4 vs 5, `q1_injection` 2 vs 1). Every strong and weak answer was scored exactly. This was a single run on 10 cases, so treat it as a sanity check rather than a benchmark (see Limitations). I also wired up Gemini, but it returned a temporary "high demand" error from Google during the eval, so no Gemini results are reported.

## Limitations

- **Small eval set, one labeller.** 10 cases labelled by me is a sanity check, not proof. A real evaluation needs many answers marked by several teachers, with agreement between them measured.
- **Retrieval can miss a criterion.** If a criterion isn't in the top-k, it isn't graded and the max score comes out too low. A fix would be to retrieve whole question sections, using the section metadata.
- **The LLM still judges.** Post-processing catches invented citations and bad arithmetic, but it can't catch a wrong judgement on a valid criterion. Teachers should review the scores, not rely on them blindly.
- **Document formats.** `.md`, `.txt` and text-based `.pdf` only. Scanned PDFs (images) and `.docx` aren't supported, and PDF line numbers refer to the extracted text.
- **Criterion detection relies on "(N points)".** Rubrics that don't state points per criterion fall back to paragraph chunks without criterion names.
- **No auth, single user.** Anyone who can reach the API can upload or grade. Fine for a local demo, not for deployment.
- **Prompt injection is mitigated, not solved.** The answer is clearly delimited and the model is told to ignore instructions inside it, but a determined attack could still sway a score.
