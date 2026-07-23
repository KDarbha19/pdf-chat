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
@app.route('/')
def index():
    #Give each browser session a Unique ID
    if 'session_id' not in session:
        import uuid
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files['pdf']
    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({"error": "Please upload a PDF file"}), 400

    session_id = session.get('session_id', 'default')

    #save pdf to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        pages, chunks = process_pdf(tmp_path, session_id)
        os.unlink(tmp_path) #delete temp file after processing
        return jsonify({
            'message' : f'PDF processed - {pages} pages, {chunks} chunks ready',
            'pages' : pages,
            'chunks' : chunks
        })
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({'error' : f'Processing failed: {str(e)}'}), 500
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '').strip()
    session_id = session.get('session_id', 'default')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        answer = get_answer(question, session_id)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Failed to get answer: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Clear the current session's PDF and chat history"""
    session_id = session.get('session_id', 'default')
    vector_stores.pop(session_id, None)
    chat_histories.pop(session_id, None)
    return jsonify({'message': 'Session cleared'})

if __name__ == '__main__':
    app.run(debug=True)