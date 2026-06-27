"""
main.py  —  Pakistan Constitution Q&A
FastAPI backend with RAG (ChromaDB) + Ollama (llama3.1:8b) + streaming
"""

import os
import re
import json
import requests
from typing import Optional, List, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\pakistan_constitution_llm"
CHROMA_DIR  = os.getenv("CHROMA_DIR",  os.path.join(BASE_DIR, "chroma_db"))
COLLECTION  = os.getenv("COLLECTION",  "pakistan_constitution")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
OLLAMA_MODEL= os.getenv("OLLAMA_MODEL","llama3.1:8b")
TOP_K       = int(os.getenv("TOP_K",   "4"))

# ── Pydantic ────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3

class AnswerResponse(BaseModel):
    answer: str
    question: str
    sources: List[str]
    suggestions: List[str]

# ── Globals ─────────────────────────────────────────────────────────────
collection = None

# ── Suggestions ─────────────────────────────────────────────────────────
SUGGESTION_MAP = {
    "arrest":    ["What is the right to fair trial?", "Can police detain me without charges?", "What is Article 10?"],
    "detention": ["What are safeguards against arrest?", "How long can police hold me?", "What is habeas corpus?"],
    "speech":    ["Can government ban social media?", "What are limits on press freedom?", "What is Article 19A?"],
    "religion":  ["Can I change my religion?", "Are religious schools protected?", "What is Article 20?"],
    "property":  ["Can government seize my land?", "What is fair compensation?", "What is Article 24?"],
    "education": ["Is education free in Pakistan?", "What age is compulsory?", "What is Article 25A?"],
    "president": ["How is the President elected?", "Can President dissolve Parliament?", "What is impeachment?"],
    "parliament":["How many seats in National Assembly?", "What is the Senate?", "How long does Assembly last?"],
    "election":  ["Who is Chief Election Commissioner?", "How are elections conducted?", "What is ECP?"],
    "emergency": ["Who can declare emergency?", "What happens to rights?", "What is Article 232?"],
    "judge":     ["How are judges appointed?", "What is Supreme Court jurisdiction?", "Can judges be removed?"],
    "amendment": ["What is 18th Amendment?", "How is Constitution changed?", "What is Article 238?"],
    "islamic":   ["What is Council of Islamic Ideology?", "How does Sharia apply?", "What is Article 227?"],
    "minority":  ["What protections for minorities?", "What is Article 36?", "Can minorities practice freely?"],
    "women":     ["What rights do women have?", "Is there gender equality?", "What is Article 34?"],
}

def get_suggestions(question: str, answer: str) -> List[str]:
    text = (question + " " + answer).lower()
    for kw, sug in SUGGESTION_MAP.items():
        if kw in text:
            return sug
    arts = re.findall(r'Article\s+(\d+[A-Z]?)', answer, re.IGNORECASE)
    if arts:
        return [f"Tell me more about Article {a}" for a in list(dict.fromkeys(arts))[:3]]
    return ["What are fundamental rights?", "How is PM elected?", "What is Supreme Court?"]

# ── RAG ─────────────────────────────────────────────────────────────────
ARTICLE_HINTS = {
    "25":  "equality citizens equal before law discrimination sex",
    "25A": "right education free compulsory children five sixteen",
    "9":   "security person liberty life",
    "10":  "arrest detention safeguards",
    "10A": "fair trial",
    "19":  "freedom speech expression press",
    "19A": "right information",
    "20":  "freedom religion",
    "23":  "property rights",
    "24":  "property compensation acquisition",
    "232": "emergency proclamation suspend",
    "63":  "disqualification member parliament",
    "90":  "federal government cabinet prime minister",
    "179": "supreme court chief justice retirement",
}

def expand_query(question: str) -> str:
    art = re.search(r'article\s*(\d+[a-zA-Z]?)', question, re.IGNORECASE)
    if art:
        hint = ARTICLE_HINTS.get(art.group(1).upper(), "")
        if hint:
            return f"{question} {hint}"
    return question

def retrieve_context(question: str) -> tuple[str, List[str]]:
    if collection is None:
        return "", []

    expanded = expand_query(question)
    results = collection.query(
        query_texts=[expanded],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    relevant = [
        (doc, meta)
        for doc, meta, dist in zip(docs, metadatas, distances)
        if dist < 0.8
    ]
    if not relevant:
        relevant = [(docs[0], metadatas[0])]

    context = "\n\n---\n\n".join(doc for doc, _ in relevant)
    sources = list(dict.fromkeys(
        meta.get("article", "")
        for _, meta in relevant
        if meta.get("article", "")
    ))
    return context, sources

# ── Prompt ───────────────────────────────────────────────────────────────
def build_prompt(question: str, context: str) -> str:
    if context:
        return f"""You are a legal assistant for the Constitution of Pakistan.

Use ONLY the constitutional text below to answer. Quote the relevant clause, then explain briefly. If not covered, say so.

Constitutional Text:
{context}

Question: {question}

Answer:"""
    else:
        return f"""You are a legal assistant for the Constitution of Pakistan. Answer accurately and cite specific Articles.

Question: {question}

Answer:"""

# ── Ollama call (non-streaming) ───────────────────────────────────────────
def ollama_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()

# ── Ollama streaming ──────────────────────────────────────────────────────
async def stream_generate(
    question: str,
    max_tokens: int,
    temperature: float,
) -> AsyncGenerator[str, None]:

    context, rag_sources = retrieve_context(question)
    prompt = build_prompt(question, context)

    full_text = ""

    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        },
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    full_text += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                if data.get("done"):
                    break
            except Exception:
                continue

    # Final metadata
    article_sources = [
        f"Article {a}"
        for a in re.findall(r'Article\s+(\d+[A-Z]?)', full_text, re.IGNORECASE)
    ]
    all_sources  = list(dict.fromkeys(rag_sources + article_sources))[:6]
    suggestions  = get_suggestions(question, full_text)

    yield f"data: {json.dumps({'sources': all_sources, 'suggestions': suggestions})}\n\n"
    yield "data: [DONE]\n\n"

# ── ChromaDB loader ───────────────────────────────────────────────────────
def load_chroma():
    global collection
    if not os.path.exists(CHROMA_DIR):
        print("WARNING: ChromaDB not found. Run ingest.py first.")
        return
    print(f"Loading ChromaDB from {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    try:
        collection = client.get_collection(name=COLLECTION, embedding_function=ef)
        print(f"✓ ChromaDB ready — {collection.count()} chunks")
    except Exception as e:
        print(f"WARNING: ChromaDB error: {e}")

# ── Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*55)
    print("Pakistan Constitution Q&A — Ollama + RAG + Streaming")
    print("="*55)
    load_chroma()

    # Check Ollama is running
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"✓ Ollama running — models: {models}")
    except Exception:
        print("WARNING: Ollama not reachable at", OLLAMA_URL)

    yield
    print("Shutting down...")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pakistan Constitution Q&A", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model": OLLAMA_MODEL,
        "rag": collection is not None,
        "chunks": collection.count() if collection else 0,
    }

@app.get("/health")
def health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {
        "status": "healthy",
        "ollama": ollama_ok,
        "rag": collection is not None,
    }

@app.post("/ask/stream")
async def ask_stream(req: QuestionRequest):
    return StreamingResponse(
        stream_generate(req.question, req.max_tokens, req.temperature),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    context, rag_sources = retrieve_context(req.question)
    prompt  = build_prompt(req.question, context)
    answer  = ollama_generate(prompt, req.max_tokens, req.temperature)

    article_sources = [f"Article {a}" for a in re.findall(r'Article\s+(\d+[A-Z]?)', answer, re.IGNORECASE)]
    all_sources = list(dict.fromkeys(rag_sources + article_sources))[:6]

    return AnswerResponse(
        answer=answer,
        question=req.question,
        sources=all_sources,
        suggestions=get_suggestions(req.question, answer),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)