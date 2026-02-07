import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings # <--- LOCAL
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DB_DIR = "hcmakers_db"

# 1. SETUP LOCAL MEMORY
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. SETUP BRAIN (Gemini Flash for speed/free)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

system_prompt = (
    "You are the HC Makers Assistant. "
    "Answer using ONLY the Context below. "
    "If the answer is not there, say 'I cannot find that info on the website.' "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

@app.route('/chat', methods=['POST'])
def chat():
    if not os.path.exists(DB_DIR):
        return jsonify({"response": "Database booting up..."})

    # Retrieve answer
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    
    try:
        user_msg = request.json.get('message', '')
        response = chain.invoke({"input": user_msg})
        return jsonify({"response": response['answer']})
    except Exception as e:
        return jsonify({"response": "Sorry, I am having trouble connecting right now."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)