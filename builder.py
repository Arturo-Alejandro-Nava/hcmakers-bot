import os
import requests
import shutil
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- UPDATED IMPORTS (OFFLINE MEMORY) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIG ---
DB_DIR = "hcmakers_db"
START_URL = "https://www.hcmakers.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_website_data():
    print(f"--- SCANNING: {START_URL} ---")
    
    docs = []
    # 1. Scrape Homepage
    try:
        r = requests.get(START_URL, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.get_text(" ", strip=True)
        docs.append(Document(page_content=text, metadata={"source": START_URL}))
        
        # 2. Find PDFs (Look everywhere)
        pdf_links = set()
        for a in soup.find_all('a', href=True):
            full_link = urljoin(START_URL, a['href'])
            if ".pdf" in full_link.lower():
                pdf_links.add(full_link)
        
        print(f"  > Found {len(pdf_links)} PDFs.")

        # 3. Download & Read PDFs
        if pdf_links:
            import tempfile
            print("  > Downloading PDFs...")
            for pdf in pdf_links:
                try:
                    # Download to temp file
                    r_pdf = requests.get(pdf, headers=HEADERS)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(r_pdf.content)
                        temp_path = f.name
                    
                    # Read
                    loader = PyPDFLoader(temp_path)
                    pdf_docs = loader.load()
                    for d in pdf_docs: d.metadata["source"] = pdf
                    docs.extend(pdf_docs)
                    
                    # Clean up
                    os.remove(temp_path)
                except Exception as e:
                    print(f"    x Skipped {pdf}: {e}")

    except Exception as e:
        print(f"Website Scan Error: {e}")
        
    return docs

def build_knowledge():
    docs = get_website_data()
    
    if not docs:
        print("❌ CRITICAL: No data found. Is the internet working?")
        return

    print(f"--- SAVING TO BRAIN ({len(docs)} chunks) ---")
    print("This runs locally on your CPU (may take 30s)...")
    
    # SPLIT TEXT
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # EMBED (Using Local HuggingFace - No Google Errors)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # CLEAR OLD DB
    if os.path.exists(DB_DIR):
        try: shutil.rmtree(DB_DIR)
        except: pass

    # SAVE
    Chroma.from_documents(splits, embeddings, persist_directory=DB_DIR)
    print("✅ SUCCESS! Database built.")

if __name__ == "__main__":
    build_knowledge()