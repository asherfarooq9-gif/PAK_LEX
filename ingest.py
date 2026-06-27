"""
ingest.py — Pakistan Constitution ingestion pipeline v3
Semantic paragraph chunking — handles PDF article numbering errors gracefully.
Usage: venv/Scripts/python ingest.py
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pdfplumber

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\pakistan_constitution_llm"
PDF_PATH    = os.getenv("PDF_PATH",    os.path.join(BASE_DIR, "consitution rights.pdf"))
CHROMA_DIR  = os.getenv("CHROMA_DIR",  os.path.join(BASE_DIR, "chroma_db"))
COLLECTION  = os.getenv("COLLECTION",  "pakistan_constitution")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE  = 600
OVERLAP     = 80
# Main body: skip TOC (before pos 46838) and Schedules (after pos 383860)
BODY_START  = 46838
BODY_END    = 383860

# ── Extract ───────────────────────────────────────────────────────────
def extract_body(pdf_path: str) -> str:
    print(f"Reading PDF: {pdf_path}")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text = re.sub(r'CONSTITUTION OF PAKISTAN\s*\n?', '', text)
                pages.append(text)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{total} pages...")
    full = '\n'.join(pages)
    print(f"✓ Extracted {len(full):,} total chars")

    # Slice to main body only
    body = full[BODY_START:BODY_END]
    print(f"✓ Body (articles only): {len(body):,} chars")
    return body

# ── Clean ─────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = re.sub(r'\d+\*',    '',  text)   # footnote markers
    text = re.sub(r'\[\d+\]',  '',  text)   # [1], [2]
    text = re.sub(r'\.{4,}[\s\d]*', '', text)  # TOC dots
    text = re.sub(r'\f',       '\n', text)  # form feeds
    text = re.sub(r'[ \t]+',   ' ',  text)
    text = re.sub(r'\n{3,}',   '\n\n', text)
    return text.strip()

# ── Detect article label for a chunk ─────────────────────────────────
ART_RE = re.compile(r'\b(\d+[A-Z]?)\.\s+[A-Z]')

def detect_article(text: str) -> str:
    """Return 'Article X' label from chunk text, or 'Constitution'."""
    nums = ART_RE.findall(text[:200])
    if nums:
        return f"Article {nums[0]}"
    return "Constitution"

# ── Chunk ─────────────────────────────────────────────────────────────
def chunk(text: str) -> list[dict]:
    """
    Split on double-newline paragraph boundaries with sliding window.
    Each chunk gets an article label from its content.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) < CHUNK_SIZE:
            buffer += ("\n\n" + para) if buffer else para
        else:
            if buffer:
                chunks.append(buffer.strip())
            # If para itself is huge, split it
            if len(para) > CHUNK_SIZE:
                pos = 0
                while pos < len(para):
                    end = min(pos + CHUNK_SIZE, len(para))
                    if end < len(para):
                        for punct in ['. ', '.\n', ') ']:
                            found = para.rfind(punct, pos, end)
                            if found > pos + CHUNK_SIZE // 2:
                                end = found + 1
                                break
                    chunks.append(para[pos:end].strip())
                    next_pos = end - OVERLAP
                    pos = next_pos if next_pos > pos else pos + CHUNK_SIZE
                buffer = ""
            else:
                buffer = para

    if buffer:
        chunks.append(buffer.strip())

    # Build chunk dicts with article labels
    result = []
    for i, text in enumerate(chunks):
        if not text:
            continue
        result.append({
            "id":      f"chunk_{i}",
            "text":    text,
            "article": detect_article(text),
            "idx":     i,
        })

    print(f"✓ Created {len(result)} chunks")
    return result

# ── Store ─────────────────────────────────────────────────────────────
def store(chunks: list[dict]):
    print(f"Connecting to ChromaDB: {CHROMA_DIR}")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
        print("  Deleted old collection")
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        col.add(
            ids       = [c["id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [{"article": c["article"], "chunk_id": c["idx"]} for c in batch],
        )
        print(f"  Embedded {min(i+BATCH, len(chunks))}/{len(chunks)}...")

    print(f"✓ Stored {col.count()} chunks in '{COLLECTION}'")
    return col

# ── Verify ────────────────────────────────────────────────────────────
def verify(col):
    tests = [
        "What is Article 25 equality of citizens?",
        "What happens during emergency?",
        "How is Prime Minister elected?",
    ]
    print("\n── Verification ────────────────────────────")
    for q in tests:
        r = col.query(query_texts=[q], n_results=1)
        doc  = r["documents"][0][0][:120]
        meta = r["metadatas"][0][0]["article"]
        print(f"Q: {q}")
        print(f"   [{meta}] {doc}...\n")

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("Pakistan Constitution — Ingestion v3")
    print("="*55 + "\n")

    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found:\n  {PDF_PATH}")
        exit(1)

    body   = extract_body(PDF_PATH)
    text   = clean(body)
    chunks = chunk(text)
    col    = store(chunks)
    verify(col)

    print("✓ Ingestion complete. Start server with:\n  venv\\Scripts\\python -m uvicorn main:app --reload --port 8000\n")