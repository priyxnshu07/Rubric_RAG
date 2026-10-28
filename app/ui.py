"""Streamlit front end. It talks to the FastAPI backend over HTTP.

Run the API first (uvicorn app.main:app), then: streamlit run app/ui.py
"""
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RubricRAG", layout="wide")
st.title("RubricRAG")
st.caption("Rubric-grounded feedback: every judgement cites the rubric criteria it is based on.")


def api(method: str, path: str, **kwargs):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with: uvicorn app.main:app")
        st.stop()
    if r.status_code >= 400:
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    return r.json()


DEFAULT_QUESTION = "Answer the task described in the rubric."

grade_tab, teacher_tab = st.tabs(["Grade an answer", "Teacher: upload a rubric"])

with grade_tab:
    rubrics = api("GET", "/rubrics")
    if not rubrics:
        st.info("No rubrics yet. Upload one in the Teacher tab, or run: python -m app.ingest data/<folder>")
    else:
        chosen = st.selectbox("Rubric", rubrics)
        question = st.text_input("Question", value=DEFAULT_QUESTION)
        answer = st.text_area("Student answer", height=200, placeholder="Paste the student's answer here")
        if st.button("Grade", type="primary", disabled=not (question.strip() and answer.strip())):
            with st.spinner("Finding the matching rubric criteria and grading (twice, for consistency)..."):
                res = api("POST", "/grade", json={"rubric_collection": chosen, "question": question,
                                                  "answer": answer})
                chunks = {c["id"]: c for c in api("GET", f"/rubrics/{chosen}/chunks")}

            if res["needs_human_review"]:
                st.error("### ⚠️ Needs teacher review\n" + "\n".join(f"- {r}" for r in res["reasons"]))
            else:
                st.success("No review flags: the checks found nothing unusual.")

            st.metric("Score", f"{res['score']} / {res['max_score']}")
            st.write(res["overall_feedback"])

            st.subheader("Feedback per criterion")
            for c in res["criteria"]:
                with st.container(border=True):
                    st.markdown(f"**{'✅' if c['met'] else '❌'} {c['name']}**")
                    st.write(c["feedback"])
                    cols = st.columns(max(len(c["cited_chunk_ids"]), 1))
                    for col, cid in zip(cols, c["cited_chunk_ids"]):
                        ch = chunks.get(cid)
                        if not ch:
                            continue
                        with col.popover(f"📄 Rubric {cid}", help="Show the rubric text this is based on"):
                            st.caption(f"{ch['criterion'] or 'General'} · {ch['source']}, "
                                       f"lines {ch['start_line']}-{ch['end_line']}")
                            st.text(ch["text"])
                    if not c["cited_chunk_ids"]:
                        st.caption("No rubric citation for this criterion.")

with teacher_tab:
    rubric_id = st.text_input("Rubric name", placeholder="e.g. photosynthesis-q1")
    rubric_file = st.file_uploader("Rubric (.md, .txt or .pdf)", type=["md", "txt", "pdf"])
    notes_file = st.file_uploader("Course notes (optional, .md, .txt or .pdf)", type=["md", "txt", "pdf"])
    if st.button("Upload", type="primary", disabled=not (rubric_id and rubric_file)):
        files = {"rubric": (rubric_file.name, rubric_file.getvalue())}
        if notes_file:
            files["notes"] = (notes_file.name, notes_file.getvalue())
        res = api("POST", "/ingest/files", data={"rubric_id": rubric_id}, files=files)
        st.success(f"Stored '{res['rubric_id']}': {res['rubric_chunks']} rubric chunks, "
                   f"{res['notes_chunks']} notes chunks.")
