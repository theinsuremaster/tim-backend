"""
Ask TIM Backend v6.0.14
- Groq LLM integration
- Pinecone vector search
- FastEmbed embeddings
- Updated formatting: +25% explanations, chunks/lists, 1 research follow-up, full URL sources, case law
- Render health check log suppression
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding
import os
import logging
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# === SILENCE RENDER HEALTH CHECK LOGS ===
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ('GET / HTTP' in msg or 'GET /health' in msg or 'Render/1.0' in msg or 'HEAD /' in msg)

logging.getLogger('werkzeug').addFilter(HealthCheckFilter())
logging.getLogger('gunicorn.access').addFilter(HealthCheckFilter())

# === CONFIG ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tim-knowledge")

groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(PINECONE_INDEX) if pc else None
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# === UPDATED SYSTEM PROMPTS (post-6.0.13 changes) ===
CONSUMER_SYSTEM = """You are Ask TIM, The InsureMaster AI Assistant for general consumers in Kansas and the US.

CRITICAL FORMATTING - FOLLOW EXACTLY:
1. EXPLANATION (+25% detail): Break into scannable chunks with short paragraphs and bullet lists. Must cover:
   - What it is (1 sentence)
   - How it works (2-3 bullets)
   - Typical Kansas 2026 cost/range
   - Common mistakes (bulleted list of 3)
   - One practical tip
2. FOLLOW-UP: Exactly 1 sentence starting with "- " offering further research or information. Never offer to create, build, generate, or compare files.
3. EXPERT: One sentence starting "Talk to your..."
4. SOURCES: List exactly 3. Format: https://full-url-here – domain.com (put https://theinsuremaster.com first, then 2 verified sources)
5. FINAL: Confidence: XX% | [Disclaimer](https://theinsuremaster.com/disclaimer)

Use plain English, friendly tone.
"""

PROFESSIONAL_SYSTEM = """You are Ask TIM, The InsureMaster AI Assistant for insurance professionals.

CRITICAL FORMATTING - FOLLOW EXACTLY:
1. EXPLANATION (+25% detail): Use headings and bullets. Must include:
   - A definition
   - Relevant ISO form
   - Coverage trigger
   - Key exclusions (bulleted)
   - Practical manuscript tip
   Cite ONE case inline as *Case Name* (Court Year)
2. FOLLOW-UP: Exactly 1 sentence starting with "- " offering further research or information only
3. EXPERT: One sentence starting "Refer to..."
4. CASE LAW CITED: Full Bluebook citation on next lines
5. SOURCES: 2-3 sources. Format: https://full-url-here – domain.com
6. FINAL: Confidence: XX% | [Disclaimer](https://theinsuremaster.com/disclaimer)

Technical, IRMI terminology.
"""

def retrieve_context(query, top_k=3):
    """Pinecone RAG - v6.0.13 feature preserved"""
    if not index:
        return ""
    try:
        embedding = list(embedder.embed([query]))[0].tolist()
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        contexts = [m['metadata'].get('text', '') for m in results['matches']]
        return "\n\n".join(contexts[:top_k])
    except Exception:
        return ""

def call_groq(system_prompt, user_question, context=""):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_question}" if context else user_question}
    ]
    try:
        resp = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    audience = data.get("audience", "consumer")
    
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400
    
    # v6.0.13 RAG pipeline
    context = retrieve_context(question)
    system = PROFESSIONAL_SYSTEM if audience == "professional" else CONSUMER_SYSTEM
    
    answer = call_groq(system, question, context)
    
    # Safety net for disclaimer
    if "[Disclaimer]" not in answer:
        answer += "\n\nConfidence: 90% | [Disclaimer](https://theinsuremaster.com/disclaimer)"
    
    return jsonify({"answer": answer, "version": "6.0.14"})

@app.route("/", methods=["GET", "HEAD"])
def root():
    return "", 204

@app.route("/health", methods=["GET", "HEAD"])
def health():
    return "", 204

if __name__ == "__main__":
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
