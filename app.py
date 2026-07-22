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