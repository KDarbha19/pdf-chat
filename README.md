UPLOAD PDF

LOAD - Extract text from PDF

SPLIT - break text into chunks(500 words each)

EMBED - Convert each chunk into a vector (list of numbers) that represent its meaning

STORE - Save all vectors in FAISS vector database

ASK QUESTION

RETRIEVE - Convert question into vector, find 3 most similar chunks in the database

GENERATE - Send question + relevant chunks to LLama, get answer back(this lets it work with huge documents)

ANSWER

RAG