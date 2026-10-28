"""Streamlit front end. It talks to the FastAPI backend over HTTP.

Run the API first (uvicorn app.main:app), then: streamlit run app/ui.py
"""
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RubricRAG", layout="wide")
st.title("RubricRAG")
st.caption("Rubric-grounded feedback: every score cites the rubric lines it is based on.")


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


teacher, student = st.tabs(["1. Teacher: upload rubric", "2. Grade an answer"])

with teacher:
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

with student:
    rubrics = api("GET", "/rubrics")
    if not rubrics:
        st.info("No rubrics yet. Upload one in the Teacher tab.")
    else:
        chosen = st.selectbox("Rubric", rubrics)
        question = st.text_input("Question")
        answer = st.text_area("Student answer", height=200)
        if st.button("Grade", type="primary", disabled=not (question.strip() and answer.strip())):
            with st.spinner("Retrieving rubric chunks and grading..."):
                res = api("POST", "/grade", json={"rubric_collection": chosen, "question": question,
                                                  "answer": answer})
                chunks = {c["id"]: c for c in api("GET", f"/rubrics/{chosen}/chunks")}

            st.metric("Score", f"{res['score']} / {res['max_score']}")
            if res["needs_human_review"]:
                st.warning("Needs human review:\n\n" + "\n".join(f"- {r}" for r in res["reasons"]))
            st.write(res["overall_feedback"])
            for c in res["criteria"]:
                with st.expander(f"{'✅' if c['met'] else '❌'} {c['name']}", expanded=True):
                    st.write(c["feedback"])
                    for cid in c["cited_chunk_ids"]:
                        ch = chunks.get(cid)
                        if ch:
                            st.caption(f"[{cid}] {ch['criterion'] or 'General'} "
                                       f"({ch['source']}, lines {ch['start_line']}-{ch['end_line']})")
                            st.text(ch["text"])
