# backend.py - TheInsureMaster v6.0.1
# Stack: Flask 3.0.3, Groq 0.9.0, Pinecone 5.0.1, FastEmbed 0.2.2

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding
import fitz # PyMuPDF
from docx import Document
from PIL import Image
import requests
from bs4 import BeautifulSoup
import httpx

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "insuremaster-v6")

groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# FastEmbed local embedding (no API calls)
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

SYSTEM_PROMPT = """You are TheInsureMaster v6.0.1. Answer in 180-220 words.
Use only sources: iii.org, bankrate.com, investopedia.com, irmi.com, wiley.law,
pillsburylaw.com, naic.org, biba.org.uk, fca.org.uk, lloyds.com, ambest.com,
lanacion.com.ar, argentina.gob.ar
Cite with superscripts ¹²³. List caselaw as text only. End with disclaimer.
Confidence: 0.xx | v6.0.1"""

def embed_text(text: str):
    """Generate embedding using FastEmbed"""
    embeddings = list(embedder.embed([text]))
    return embeddings[0].tolist()

def search_knowledge(query: str, top_k=5):
    """Search Pinecone for relevant sources"""
    vector = embed_text(query)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results.matches:
        meta = match.metadata
        contexts.append(f"{meta.get('title')} — {meta.get('domain')}")
    return contexts

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v6.0.1"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    jurisdiction = data.get("jurisdiction", "US")

    # 1. Retrieve context
    sources = search_knowledge(question)
    context_str = "\n".join(sources)

    # 2. Build prompt
    user_prompt = f"""Question: {question}
Jurisdiction: {jurisdiction}
Relevant sources: {context_str}
Provide answer with superscripts, sources list, and disclaimer."""

    # 3. Call Groq
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "answer": answer,
        "sources_used": sources,
        "jurisdiction": jurisdiction,
        "version": "v6.0.1"
    })

@app.route("/ingest", methods=["POST"])
def ingest():
    """Ingest PDF/DOCX/URL into Pinecone"""
    if "file" in request.files:
        file = request.files["file"]
        text = ""
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        elif file.filename.endswith(".docx"):
            doc = Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])

        vector = embed_text(text[:2000])
        index.upsert([(file.filename, vector, {"title": file.filename, "domain": "upload"})])
        return jsonify({"status": "ingested", "chars": len(text)})

    if "url" in request.json:
        url = request.json["url"]
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text()[:5000]
        vector = embed_text(text[:2000])
        index.upsert([(url, vector, {"title": soup.title.string, "domain": url})])
        return jsonify({"status": "ingested", "url": url})

    return jsonify({"error": "no file or url"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)#
