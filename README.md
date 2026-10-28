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
   │ Chunking             │ ─────────────────▶ │ ChromaDB (local)     │   │ Groq LLM        │
   │ rubric: 1 per line   │                    │ filtered by          │   │ JSON output,    │
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
| Vector store | ChromaDB (local, persistent) | `app/store.py` |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | `app/store.py` |
| LLM | Groq API (`llama-3.3-70b-versatile` by default) | `app/grader.py` |
| Schemas | Pydantic | `app/schemas.py` |
| UI | Streamlit | `app/ui.py` |
| Tests | pytest | `tests/` |

```
app/     API, chunking, vector store, grader, UI
data/    sample rubric (3 questions), course notes, example answers
eval/    labelled test cases + evaluation script
tests/   pytest unit and integration tests (no API key needed)
```

## How it works

1. **Ingest.** The rubric is split **one chunk per non-empty line**, because each rubric line is a self-contained rule and citing a single line is the most useful feedback. Notes are split into paragraphs. Each chunk keeps its line numbers and an id like `R9` (rubric line 9) or `N4-6` (notes lines 4–6).
2. **Embed and store.** Chunks are embedded with `all-MiniLM-L6-v2` and stored in ChromaDB with metadata (`rubric_id`, `doc_type`, line numbers). Re-uploading a rubric replaces its old chunks.
3. **Retrieve.** For a student answer, the question and answer text form the query. The top-k rubric lines (default 8) and top-3 notes paragraphs for that `rubric_id` are retrieved. One rubric can cover a whole quiz, so retrieval is what picks out the right question's criteria.
4. **Grade.** The LLM receives the chunks tagged with their ids, the question, and the answer wrapped in `<answer>` tags with an instruction to ignore any instructions inside it. It returns JSON: one entry per criterion with score, max score, feedback, and cited ids. This is validated with Pydantic and retried once if malformed.
5. **Verify in code.** Citations that weren't in the retrieved context are removed, scores are clamped to each criterion's max, and the total is summed in Python. Anything suspicious is returned as a `warning` instead of being hidden.

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY (free at console.groq.com)
```

### Run

```bash
uvicorn app.main:app --reload          # API at http://localhost:8000 (docs at /docs)
streamlit run app/ui.py                # UI at http://localhost:8501 (second terminal)
```

In the UI: upload `data/biology_quiz_rubric.txt` and `data/biology_notes.txt` as `biology-quiz-1`, then paste an answer from `data/sample_answers.txt`.

### Test

```bash
pytest
```

The tests use a fake LLM, so they need no API key. They do use the real embedding model and a temporary ChromaDB.

## Evaluation

`eval/cases.jsonl` holds 10 student answers across the 3 quiz questions (strong, partial and weak for each, plus one prompt-injection attempt). I labelled each with a reference score by applying the rubric by hand.

```bash
python -m eval.run_eval --retrieval-only   # retrieval metrics only, no API key
python -m eval.run_eval                    # full pipeline, needs GROQ_API_KEY
```

| Metric | What it checks |
|---|---|
| Retrieval recall@k | Did retrieval return all of the question's rubric lines? |
| Score MAE / within 1 point | How close the LLM's total is to my reference score |
| Max score correct | Did the LLM grade the right question's criteria (max = 10)? |
| Citation precision | Share of cited rubric lines that belong to the right question |
| Invented citations removed | How often post-processing had to strip a fake citation |

**Results:** see `eval/results.md` (written by the script).

- **Retrieval:** recall@k = **1.00** for k = 4, 6 and 8 on all 10 cases: every answer retrieved all four criteria for its own question. The default is k = 8 to leave headroom for rubrics with more criteria per question. This is an easy setting, though: each rubric line is prefixed with its question name, which makes matching easy.
- **Grading (LLM):** run `python -m eval.run_eval` with a Groq key to fill in MAE, accuracy and citation precision.

## Limitations

- **Small eval set, one labeller.** 10 cases labelled by me is a sanity check, not proof. A real evaluation needs many answers marked by several teachers, with agreement between them measured.
- **Retrieval can miss a criterion.** If a rubric line isn't in the top-k, it isn't graded and the max score comes out too low. A fix would be to retrieve whole question blocks, or to use a structured rubric format.
- **The LLM still judges.** Post-processing catches invented citations and bad arithmetic, but it can't catch a wrong judgement on a valid criterion. Teachers should review the scores, not rely on them blindly.
- **Plain text only.** Uploads must be `.txt`/`.md`; PDF/DOCX parsing isn't implemented.
- **No auth, single user.** Anyone who can reach the API can upload or grade. Fine for a local demo, not for deployment.
- **Prompt injection is mitigated, not solved.** The answer is clearly delimited and the model is told to ignore instructions inside it, but a determined attack could still sway a score.
