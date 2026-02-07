import streamlit as st
import os
import requests
import tempfile
import shutil
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

st.set_page_config(page_title="HC Makers Bot", page_icon="🤖")

DB_DIR = "hcmakers_db"
START_URL = "https://www.hcmakers.com"

# --- 1. SETUP RESOURCES (Cached to prevent crashes) ---
@st.cache_resource
def load_resources():
    # Load the AI Model ONCE and keep it ready
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

@st.cache_resource
def build_vector_db():
    # Only run this if DB is missing
    if os.path.exists(DB_DIR):
        return True

    embeddings = load_resources()
    headers = {"User-Agent": "Mozilla/5.0"}
    docs = []

    try:
        # Crawl Website
        r = requests.get(START_URL, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.get_text(" ", strip=True)
        docs.append(Document(page_content=text, metadata={"source": START_URL}))

        # Find & Download PDFs
        for a in soup.find_all('a', href=True):
            link = urljoin(START_URL, a['href'])
            if ".pdf" in link.lower():
                try:
                    r_pdf = requests.get(link, headers=headers)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(r_pdf.content)
                        name = f.name
                    loader = PyPDFLoader(name)
                    pdf_docs = loader.load()
                    for d in pdf_docs: d.metadata["source"] = link
                    docs.extend(pdf_docs)
                except: pass

        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = splitter.split_documents(docs)
            Chroma.from_documents(splits, embeddings, persist_directory=DB_DIR)
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    return False

# --- APP START ---
st.title("🤖 HC Makers Assistant")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.text_input("Enter Google API Key", type="password")
    if not api_key: st.stop()

# Initialize Everything
embeddings = load_resources()
build_vector_db()

# --- CHAT ENGINE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How can we help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Reconnect to DB using cached embeddings
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            retriever = vectorstore.as_retriever()
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            
            system_prompt = (
                "You are the HC Makers AI. Answer strictly from the Context. "
                "Context: {context}"
            )
            template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, template))
            
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {str(e)}")