import streamlit as st
import tempfile
import os
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END
from typing import TypedDict

from groq import Groq

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =========================
# INIT
# =========================
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="Agentic PDF Assistant", layout="wide")

# =========================
# UI
# =========================
st.markdown("""
<style>
    .stApp { background-color:#0e0e0e; color:white; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic PDF Assistant (LangGraph + Critic + Summarizer)")

LOGO_PATH = "logo.png"

# =========================
# UTIL
# =========================
def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# =========================
# LLM CALL
# =========================
def call_llm(prompt, system):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return res.choices[0].message.content

# =========================
# RETRIEVER
# =========================
def retrieve(db, query):
    return db.similarity_search(query, k=4)

# =========================
# ANSWER AGENT
# =========================
def generate_answer(query, context):
    prompt = f"""
Use ONLY context.

Context:
{context}

Question:
{query}

Answer:
"""
    return call_llm(prompt, "You are a helpful QA assistant.")

# =========================
# 🔥 OLD CRITIC (RESTORED FORMAT)
# =========================
def critic(answer, context, query):

    prompt = f"""
You are a strict verification agent.

TASK:
1. Verify correctness using context
2. Find exact evidence
3. Give confidence score (0-100)
4. Fix errors if needed

FORMAT:
Verdict:
Evidence:
Reasoning:
Confidence:
Final Answer:

Context:
{context}

Question:
{query}

Answer:
{answer}
"""
    return call_llm(prompt, "You are a factual verification system.")

# =========================
# SUMMARIZER (UNCHANGED)
# =========================
def summarize_chat(chat_history):

    conversation = "\n".join(
        [f"User: {c['query']}\nAssistant: {c['answer']}" for c in chat_history]
    )

    prompt = f"""
Create structured summary:

1. Overview
2. Key Questions
3. Key Answers
4. Insights
5. Final Summary

Conversation:
{conversation}
"""
    return call_llm(prompt, "You are a documentation summarizer.")

# =========================
# LANGGRAPH STATE
# =========================
class State(TypedDict):
    query: str
    context: str
    answer: str
    critique: str
    final_answer: str

# =========================
# NODES
# =========================
def retrieve_node(state):
    docs = retrieve(st.session_state.db, state["query"])
    context = "\n\n".join(clean(d.page_content) for d in docs)
    return {"context": context}

def answer_node(state):
    answer = generate_answer(state["query"], state["context"])
    return {"answer": answer}

def critic_node(state):
    critique = critic(state["answer"], state["context"], state["query"])

    # Extract final answer from critic (fallback safe)
    final_answer = state["answer"]
    if "Final Answer:" in critique:
        try:
            final_answer = critique.split("Final Answer:")[-1].strip()
        except:
            pass

    return {
        "critique": critique,
        "final_answer": final_answer
    }

def final_node(state):
    return state

# =========================
# LANGGRAPH BUILD
# =========================
graph = StateGraph(State)

graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("critic", critic_node)
graph.add_node("final", final_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "critic")
graph.add_edge("critic", "final")
graph.add_edge("final", END)

app = graph.compile()

# =========================
# PDF LOADER
# =========================
def build_db(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name

    loader = PyPDFLoader(path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

# =========================
# SESSION STATE
# =========================
if "db" not in st.session_state:
    st.session_state.db = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "summary" not in st.session_state:
    st.session_state.summary = None

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload PDF")

if file:
    st.session_state.db = build_db(file)
    st.success("PDF Loaded Successfully")

# =========================
# CHAT LOOP (LANGGRAPH)
# =========================
if st.session_state.db:

    query = st.chat_input("Ask your PDF...")

    if query:

        result = app.invoke({
            "query": query
        })

        st.session_state.chat.append({
            "query": query,
            "answer": result["final_answer"],
            "critique": result["critique"]
        })

    # =========================
    # CHAT DISPLAY
    # =========================
    for msg in st.session_state.chat:

        st.chat_message("user").write(msg["query"])
        st.chat_message("assistant").write(msg["answer"])

        with st.expander("🔬 Fact Check (Critic)"):
            st.write(msg["critique"])

    # =========================
    # SUMMARY MODULE (RESTORED)
    # =========================
    if st.session_state.chat:

        if st.button("📄 Generate Summary"):
            st.session_state.summary = summarize_chat(st.session_state.chat)

    if st.session_state.summary:

        st.subheader("📄 Chat Summary")
        st.write(st.session_state.summary)

        # =========================
        # PDF EXPORT
        # =========================
        if st.button("📥 Export PDF Report"):

            file_path = "agentic_report.pdf"
            doc = SimpleDocTemplate(file_path, pagesize=A4)

            styles = getSampleStyleSheet()
            content = []

            if os.path.exists(LOGO_PATH):
                content.append(Image(LOGO_PATH, 1.5*inch, 1.5*inch))

            content.append(Paragraph("Agentic PDF Report", styles["Title"]))
            content.append(Spacer(1, 12))

            summary = Paragraph(
                st.session_state.summary.replace("\n", "<br/>"),
                styles["BodyText"]
            )
            content.append(summary)

            doc.build(content)

            with open(file_path, "rb") as f:
                st.download_button(
                    "Download Report",
                    f,
                    file_name="Agentic_Report.pdf",
                    mime="application/pdf"
                )

else:
    st.info("Upload a PDF to start chatting")