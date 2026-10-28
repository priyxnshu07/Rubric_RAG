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
    rubric_file = st.file_uploader("Rubric (.txt or .md)", type=["txt", "md"])
    notes_file = st.file_uploader("Course notes (optional, .txt or .md)", type=["txt", "md"])
    if st.button("Upload", type="primary", disabled=not (rubric_id and rubric_file)):
        files = {"rubric": (rubric_file.name, rubric_file.getvalue())}
        if notes_file:
            files["notes"] = (notes_file.name, notes_file.getvalue())
        res = api("POST", "/ingest/files", data={"rubric_id": rubric_id}, files=files)
        st.success(f"Stored '{res['rubric_id']}': {res['rubric_chunks']} rubric lines, "
                   f"{res['notes_chunks']} notes chunks.")

with student:
    rubrics = api("GET", "/rubrics")
    if not rubrics:
        st.info("No rubrics yet. Upload one in the Teacher tab.")
    else:
        chosen = st.selectbox("Rubric", rubrics)
        question = st.text_input("Question (optional)")
        answer = st.text_area("Student answer", height=200)
        if st.button("Grade", type="primary", disabled=not answer.strip()):
            with st.spinner("Retrieving rubric lines and grading..."):
                res = api("POST", "/grade", json={"rubric_id": chosen, "question": question, "answer": answer})

            st.metric("Score", f"{res['total_score']:g} / {res['max_score']:g}")
            for w in res["warnings"]:
                st.warning(w)
            for c in res["criteria"]:
                with st.expander(f"{c['criterion']}: {c['score']:g} / {c['max_score']:g}", expanded=True):
                    st.write(c["feedback"])
                    if c["citations"]:
                        st.caption("Based on: " + ", ".join(c["citations"]))
            st.subheader("Cited rubric lines")
            for cit in res["citations"]:
                lines = (f"line {cit['start_line']}" if cit["start_line"] == cit["end_line"]
                         else f"lines {cit['start_line']}-{cit['end_line']}")
                st.markdown(f"**[{cit['id']}]** ({lines}) {cit['text']}")
