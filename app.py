import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template, session
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
app.secret_key = "pdfchat_secret_key"

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "mps"}
)

print("Connecting to Ollama...")
llm = OllamaLLM(model="llama3.2:3b")
print("Ready")

vector_stores = {}
chat_histories = {}

#pdf processing
def process_pdf(pdf_path, session_id):
    """
    Load a pdf, split into chunks, embed them, store in FAISS
    """
    #Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    #Split into chunks: Recursive CharacterTextSplitter splits on paragraphs first then sentences etc...
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    #Embed chunks and store in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)

    #Store in memory keyed by session ID
    vector_stores[session_id] = vector_store
    chat_histories[session_id] = []

    return len(documents), len(chunks)

#RAG CHAIN
def get_answer(question, session_id):
    """
    Given a question and Session ID, retrieve relevant chunks and generate answer
    """

    if session_id not in vector_stores:
        return "Please upload a PDF first!"

    vector_store = vector_stores[session_id]
    chat_history = chat_histories[session_id]

    #Retrieve top 3 most relevant chunks 
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_chunks = retriever.invoke(question)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    #Build prompt and generate answer, including chat history so the model remebers previous questions
    history_text = ""
    for msg in chat_history[-6:]: #last 3 exchanges(6 messages)
        role = "Human" if msg["role"] == "human" else "Assistant"
        history_text += f"{role} : {msg['content']}\n"

    prompt = f""" You are a helpful assistant that answers questions about documents.
Use only the context provided to answer the question. If the answer is not in the context, say
"I Couldn't find that in the document."

CONTEXT FROM DOCUMENT:
{context}

CONVERSATION HISTORY:
{history_text}
QUESTION: {question}

ANSWER:"""

    answer = llm.invoke(prompt)

    #Save to chat history 
    chat_histories[session_id].append({"role": "human", "content": question})
    chat_histories[session_id].append({"role": "assistant", "content": answer})

    return answer

#ROUTES