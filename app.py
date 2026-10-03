import os

import streamlit as st
from groq import Groq

from agent import AgentContext, run_agent
from rag import DocumentStore, extract_text

st.set_page_config(page_title="Career Coach", page_icon="🎯", layout="wide")


def get_secret(name, default=None):
    """Env var first, then Streamlit secrets (which may not exist locally)."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


API_KEY = get_secret("GROQ_API_KEY")
MODEL = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

if not API_KEY:
    st.error(
        "No Groq API key found. Either set the `GROQ_API_KEY` environment variable, "
        "or create `.streamlit/secrets.toml` containing `GROQ_API_KEY = \"gsk_...\"`."
    )
    st.stop()

if "store" not in st.session_state:
    st.session_state.store = DocumentStore()
    st.session_state.api_messages = []  # full tool-use history sent to the model
    st.session_state.chat = []  # (role, text, trace) for display

ctx = AgentContext(
    client=Groq(api_key=API_KEY), model=MODEL, store=st.session_state.store
)

# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("📄 Your documents")
    resume = st.file_uploader("Resume (PDF or TXT)", type=["pdf", "txt", "md"])
    jd = st.text_area("Paste the job description", height=200)
    if st.button("Index documents", type="primary"):
        n = 0
        if resume:
            n += st.session_state.store.add("resume", extract_text(resume.name, resume.getvalue()))
        if jd.strip():
            n += st.session_state.store.add("job_description", jd)
        st.success(f"Indexed {n} chunks.") if n else st.warning("Add a resume or job description first.")
    st.caption(f"Indexed: {', '.join(sorted(st.session_state.store.sources)) or 'nothing yet'}")
    st.divider()
    st.caption(f"Model: `{MODEL}`")

# ---------- Main ----------
st.title("🎯 Career Coach")
st.caption("RAG over your resume + job description, with an agent that can search the web and generate interview prep.")

quick = None
c1, c2, c3 = st.columns(3)
if c1.button("Skill-gap analysis", use_container_width=True):
    quick = "Do a skill-gap analysis of my resume against the job description."
if c2.button("Mock interview questions", use_container_width=True):
    quick = "Generate interview questions for this role, tailored to my resume."
if c3.button("Improve my resume", use_container_width=True):
    quick = "Suggest specific improvements to my resume for this job, with rewritten bullet examples."

for role, text, trace in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
        if trace:
            with st.expander(f"🔧 Agent steps ({len(trace)})"):
                for t in trace:
                    st.markdown(f"**{t['tool']}** `{t['input']}`")
                    st.code(t["output"])

prompt = st.chat_input("Ask about your career, resume, or interview prep...") or quick
if prompt:
    st.session_state.chat.append(("user", prompt, None))
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.api_messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, trace = run_agent(ctx, st.session_state.api_messages)
            except Exception as e:
                reply, trace = f"Something went wrong: `{e}`", []
        st.markdown(reply)
    st.session_state.chat.append(("assistant", reply, trace))
    st.rerun()
