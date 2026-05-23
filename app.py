from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Flask app setup
app=Flask(__name__)

# Folder where uploaded PDFs will be stored
UPLOAD_FOLDER="uploads"
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables
chunks=[]
index=None

# Load embedding model
model=SentenceTransformer("all-MiniLM-L6-v2")

# Groq client setup
client=Groq(
    api_key="API"
)

# Load text from uploaded PDF
def load_pdf(file_path):
    reader=PdfReader(file_path)
    text=""
    
    # Extract text page by page
    for page in reader.pages:
        text+=(page.extract_text() or "")+"\n"
    
    return text

# Split text into chunks
def split_text_words(text,chunk_size=200,overlap=15):
    words=text.split()
    chunks=[]
    start=0
    
    while start<len(words):
        end=start+chunk_size
        chunk=words[start:end]
        
        # Store chunk
        chunks.append(" ".join(chunk))
        
        # Move with overlap
        start+=chunk_size-overlap
    
    return chunks

# Process uploaded PDF
def process_pdf(file_path):
    global chunks,index
    
    # Extract text
    text=load_pdf(file_path)
    
    # Chunk text
    chunks=split_text_words(text)
    
    if not chunks:
        raise ValueError("No text extracted from PDF")
    
    # Convert chunks to embeddings
    embeddings=model.encode(chunks)
    embeddings=np.array(embeddings).astype("float32")
    
    # Fix dimensions if needed
    if embeddings.ndim!=2:
        embeddings=embeddings.reshape(
            embeddings.shape[0],
            -1
        )
    
    # Create FAISS vector index
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    
    # Store embeddings
    index.add(  # type: ignore
        np.ascontiguousarray(
            embeddings
        )
    )

# Send prompt to Groq LLM
def ask_llm(prompt):
    
    try:
        
        response=client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        
        print("Groq Error:",str(e))
        
        return f"Error: {str(e)}"

# Homepage route
@app.route("/")
def home():
    return render_template("index.html")

# Upload PDF route
@app.route("/upload",methods=["POST"])
def upload_pdf():
    
    # Check if file exists
    if "pdf" not in request.files:
        return jsonify({
            "error":"No file uploaded"
        })
    
    file=request.files["pdf"]
    
    # Check filename
    if not file.filename:
        return jsonify({
            "error":"Invalid file"
        })
    
    # Secure filename
    filename=secure_filename(
        file.filename
    )
    
    # Allow only PDF
    if not filename.lower().endswith(".pdf"):
        return jsonify({
            "error":"Please upload PDF only"
        })
    
    # Save file
    file_path=os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )
    
    file.save(file_path)
    
    # Process PDF
    process_pdf(file_path)
    
    return jsonify({
        "message":"PDF uploaded successfully"
    })

# Ask question route
@app.route("/ask",methods=["POST"])
def ask_question():
    global chunks,index
    
    # No PDF uploaded yet
    if index is None:
        return jsonify({
            "answer":"Upload PDF first"
        })
    
    # Get user question
    query=request.json.get(
        "question"
    )
    
    if not query:
        return jsonify({
            "answer":"Please ask something"
        })
    
    # Convert question to embedding
    query_embedding=model.encode(
        [query]
    )
    
    query_embedding=np.array(
        query_embedding
    ).astype("float32")
    
    # Search top matching chunks
    k=4
    
    distances,indices=index.search(  # type: ignore
        np.ascontiguousarray(
            query_embedding
        ),
        k
    )
    
    # Retrieve relevant chunks
    retrieved_chunks=[
        chunks[i]
        for i in indices[0]
    ]
    
    # Create context
    context="\n".join(
        retrieved_chunks
    )
    
    # Prompt for LLM
    prompt=f"""
You are an academic tutor.

Answer the question using ONLY the context below.

Give a detailed answer in simple language.

Rules:
1. Explain in 10-15 lines.
2. Include definition.
3. Explain working/process.
4. Include important points if available.
5. If answer is not found, say "I don't know."

Context:
{context}

Question:
{query}

Detailed Answer:
"""
    
    # Generate final answer
    answer=ask_llm(prompt)
    
    return jsonify({
        "answer":answer
    })

# Run Flask app
if __name__=="__main__":
    app.run(debug=True)
