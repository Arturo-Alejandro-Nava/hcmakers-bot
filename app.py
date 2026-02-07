import streamlit as st
import os
import requests
import shutil
import tempfile
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# AI Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- CONFIG ---
st.set_page_config(page_title="HC Makers Bot", page_icon="🤖")
DB_DIR = "hcmakers_db"
START_URL = "https://www.hcmakers.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# --- 1. THE CRAWLER (Runs only once) ---
@st.cache_resource
def initialize_knowledge_base():
    if os.path.exists(DB_DIR):
        return True # Already exists, skip building

    st.toast("System is learning... Scanning website...", icon="🕸️")
    
    docs = []
    
    # A. Scrape Website Text
    try:
        r = requests.get(START_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.get_text(" ", strip=True)
        docs.append(Document(page_content=text, metadata={"source": START_URL}))
        
        # B. Find & Download PDFs
        pdf_links = set()
        for a in soup.find_all('a', href=True):
            full_link = urljoin(START_URL, a['href'])
            if ".pdf" in full_link.lower():
                pdf_links.add(full_link)
        
        if pdf_links:
            st.toast(f"Found {len(pdf_links)} PDFs to read...", icon="📄")
            for pdf in pdf_links:
                try:
                    r_pdf = requests.get(pdf, headers=HEADERS)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(r_pdf.content)
                        tmp_name = f.name
                    loader = PyPDFLoader(tmp_name)
                    pdf_docs = loader.load()
                    for d in pdf_docs: d.metadata["source"] = pdf
                    docs.extend(pdf_docs)
                    os.remove(tmp_name)
                except: pass

    except Exception as e:
        st.error(f"Crawling Error: {e}")
        return False

    if not docs:
        st.error("Could not find any data!")
        return False

    # C. Save to Brain (Local CPU)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Use HuggingFace (Free, no API limits)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(splits, embeddings, persist_directory=DB_DIR)
    
    st.toast("Knowledge Base Created!", icon="✅")
    return True

# --- 2. THE APP INTERFACE ---
st.title("🤖 HC Makers Assistant")

# API Key Check
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.text_input("Enter Google API Key", type="password")

if not api_key:
    st.warning("Please provide an API Key to start.")
    st.stop()

# Initialize DB
initialize_knowledge_base()

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about HC Makers services..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Checking documents..."):
            try:
                # Load DB
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                retriever = vectorstore.as_retriever()
                
                # Load Gemini
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                
                system_prompt = (
                    "You are the assistant for HC Makers. "
                    "Answer strictly based on the context below. "
                    "Context: {context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
                response = chain.invoke({"input": prompt})
                answer = response['answer']
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")