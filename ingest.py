import os
import sys
import glob
import time
import pymupdf as fitz  # PyMuPDF
from dotenv import load_dotenv

# Force utf-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

def main():
    print("=" * 60, flush=True)
    print(" 🚀 TIET Policy Lens — Document Ingestion Pipeline", flush=True)
    print("=" * 60, flush=True)

    pdf_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "pdfs")
    if not os.path.exists(pdf_folder):
        print(f"❌ PDF folder not found at: {pdf_folder}")
        return

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    print(f"📁 Found {len(pdf_files)} PDF files in {pdf_folder}")

    # 1. Parse PDFs into cleaned page documents
    cleaned_pages = []
    total_raw_pages = 0

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        try:
            doc = fitz.open(pdf_path)
            total_raw_pages += len(doc)
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text").strip()
                if len(text) >= 30:  # Ignore empty/cover artifact pages
                    cleaned_pages.append({
                        "filename": fname,
                        "page": page_num,
                        "text": text
                    })
        except Exception as e:
            print(f"⚠️ Error reading PDF '{fname}': {e}")

    print(f"📄 Total PDF Pages: {total_raw_pages} | Valid Content Pages: {len(cleaned_pages)}")

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    all_chunks = []
    for p in cleaned_pages:
        doc = Document(
            page_content=p["text"],
            metadata={
                "filename": p["filename"],
                "page": p["page"],
                "source": f"{p['filename']} — Page {p['page']}"
            }
        )
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"✂️ Total Semantically Segmented Chunks Created: {len(all_chunks)}")

    # 3. Vector Database Connection (Local persistent ./qdrant_db)
    from app.rag import client, vector_store, COLLECTION_NAME

    # Re-create collection if needed
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"🔄 Re-creating existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"✅ Created collection '{COLLECTION_NAME}' (768-dim COSINE).")

    # 4. Ingest Chunks in Batches
    batch_size = 50
    total_chunks = len(all_chunks)
    print(f"⚡ Ingesting {total_chunks} chunks using Serverless HF Embeddings...")

    start_time = time.time()
    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        print(f"   --> Ingesting batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch)} chunks)...")
        vector_store.add_documents(batch)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"🎉 SUCCESS! Ingested {total_chunks} chunks into local Qdrant DB in {elapsed:.2f} seconds.")
    print("=" * 60)

if __name__ == "__main__":
    main()
