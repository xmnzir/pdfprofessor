import streamlit as st
import tempfile
import os
import re
import base64
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

# =========================
# INIT
# =========================
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="The Document Explorer", layout="wide")

# =========================
# UI
# =========================
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: "Courier New", monospace;
    }

    .block {
        border: 1px solid #222;
        padding: 12px;
        margin: 10px 0px;
    }

    .title {
        font-size: 28px;
        letter-spacing: 2px;
    }

</style>
""", unsafe_allow_html=True)

# =========================
# LOGO
# =========================
LOGO_PATH = "logo.png"

def get_logo_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = get_logo_base64(LOGO_PATH)

if logo_b64:
    st.markdown(f"""
        <div style="text-align:left;">
            <img src="data:image/png;base64,{logo_b64}" width="220">
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='title'>PDFPROFESSOR</div>", unsafe_allow_html=True)

# =========================
# UTIL
# =========================
def clean(text):
    return re.sub(r"\s+", " ", text).strip()

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
# WEB SEARCH
# =========================
def web_search(query):
    url = f"https://duckduckgo.com/html/?q={query}"
    return url  # IMPORTANT FIX: return ONLY url for clickable use

# =========================
# RAG
# =========================
def retrieve(db, query):
    return db.similarity_search(query, k=4)

def generate_answer(query, context):
    return call_llm(
        f"""
Context:
{context}

Question:
{query}

Answer ONLY from context.
""",
        "You are a precise QA assistant."
    )

def critic(answer, context, query):
    return call_llm(
        f"""
Check correctness and give confidence.

Context:
{context}

Q:
{query}

A:
{answer}

Return:
Confidence (0-100) + Final Answer
""",
        "You are a strict evaluator."
    )

def summarize_chat(chat_history):
    conversation = "\n".join(
        [f"User: {c['query']}\nBot: {c['answer']}" for c in chat_history]
    )

    return call_llm(
        f"""
Summarize:

{conversation}
""",
        "You are a summarizer."
    )

# =========================
# STATE
# =========================
class State(TypedDict):
    query: str
    context: str
    answer: str
    critique: str
    confidence: int
    final_answer: str
    web: str

# =========================
# NODES
# =========================
def retrieve_node(state):
    docs = retrieve(st.session_state.db, state["query"])
    context = "\n\n".join(clean(d.page_content) for d in docs)
    return {"context": context}

def answer_node(state):
    return {"answer": generate_answer(state["query"], state["context"])}

def critic_node(state):
    c = critic(state["answer"], state["context"], state["query"])

    m = re.search(r"Confidence\s*[:\-]?\s*(\d+)", c)
    conf = int(m.group(1)) if m else 0

    return {
        "critique": c,
        "confidence": conf,
        "final_answer": state["answer"]
    }

def web_node(state):
    return {"web": web_search(state["query"])}

def route(state):
    return "web" if state["confidence"] < 70 else "end"

# =========================
# GRAPH
# =========================
graph = StateGraph(State)

graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("critic", critic_node)
graph.add_node("web", web_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "critic")

graph.add_conditional_edges(
    "critic",
    route,
    {"web": "web", "end": END}
)

graph.add_edge("web", END)

app = graph.compile()

# =========================
# PDF DB
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
# SESSION
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

# =========================
# CHAT
# =========================
def render_block(role, content, icon):
    st.markdown(f"""
        <div class="block">
            <b>{icon} {role}</b><br><br>
            {content}
        </div>
    """, unsafe_allow_html=True)

if st.session_state.db:

    query = st.chat_input("Ask your document...")

    if query:
        result = app.invoke({"query": query})

        st.session_state.chat.append({
            "query": query,
            "answer": result.get("final_answer", ""),
            "critique": result.get("critique", ""),
            "web": result.get("web", ""),
            "confidence": result.get("confidence", 0)
        })

    for msg in st.session_state.chat:
        render_block("USER", msg["query"], "📘")
        render_block("BOT", msg["answer"], "🤖")

        with st.expander("CRITIC"):
            st.write(msg["critique"])
            st.write("Confidence:", msg["confidence"])

        if msg.get("web"):
            st.markdown(f"WEB SEARCH LINK: [OPEN RESULTS]({msg['web']})")

# =========================
# SUMMARY + FIXED PDF EXPORT
# =========================
if st.session_state.chat:

    if st.button("GENERATE SUMMARY"):
        st.session_state.summary = summarize_chat(st.session_state.chat)

    if st.session_state.summary:
        st.subheader("SUMMARY")
        st.write(st.session_state.summary)

        # =========================
        # FIXED PDF EXPORT (WORKING)
        # =========================
        if st.button("EXPORT PDF"):

            file_path = "report.pdf"
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()

            content = []

            if logo_b64:
                import io, base64
                from reportlab.platypus import Image

                logo_img = base64.b64decode(logo_b64)
                with open("temp_logo.png", "wb") as f:
                    f.write(logo_img)

                content.append(Image("temp_logo.png", width=200, height=80))

            content += [
                Paragraph("PDFPROFESSOR REPORT", styles["Title"]),
                Spacer(1, 12),
                Paragraph(st.session_state.summary.replace("\n", "<br/>"), styles["BodyText"])
            ]

            doc.build(content)

            with open(file_path, "rb") as f:
                st.download_button(
                    "DOWNLOAD PDF",
                    f,
                    file_name="PDFProfessor_Report.pdf",
                    mime="application/pdf"
                )