import os 
import json
from flask import Flask, request, jsonify, render_template, session
from langchain_community.document_loaders import pyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile 

app = Flask(__name__)
app.secret_key = 'pdfchat_secret_key'

#Model Setup
print("Loading embedding model...")
#Embeddings convert text to vectors
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-l6-v2",
    model_kwargs = {"device": "mps"}
)

print("Connecting to Ollama...")
llm = OllamaLLM(model="llama3.2:3b")
print("Model setup complete.")