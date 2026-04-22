# backend.py - TheInsureMaster v6.0.2.2
# Compatible with original Building-Ask-TIM-chatbot.html
# - Accepts mode from frontend
# - Hello works without selection
# - Returns both 'answer' and 'response' for compatibility

import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "insuremaster-v6")
PORT = int(os.getenv("PORT", 10000))

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(PINECONE_INDEX) if pc else None
embedder = None

# --- PROMPTS (same tone split) ---
SYSTEM_PROMPTS = {
    "consumer": """You are TheInsureMaster v6.0.2.2 — CONSUMER MODE.
VOICE: Warm, direct, Kansas-neighbor tone. Plain English, no jargon. Use "you".
Answer 180-220 words. Give one clear recommendation with numbers.
Sources: iii.org, bankrate.com, investopedia.com. Cite ¹²³. No caselaw.
End: Consult licensed agent. Disclaimer: https://theinsuremaster.com/disclaimer
Confidence: 0.xx | v6.0.2.2""",

    "professional": """You are TheInsureMaster v6.0.2.2 — PROFESSIONAL MODE.
VOICE: Precise, coverage-counsel tone. Use manuscript, contra proferentem, Side A.
Answer 200-240 words. Integrate 2+ cases into analysis.
Sources: irmi.com, wiley.law, pillsburylaw.com. Cite ¹²³⁴.
End: Consult broker/counsel. Disclaimer: https://theinsuremaster.com/disclaimer
Confidence: 0.xx | v6.0.2.2"""
}

PRO_TERMS = [r"e\s*&\s*o", r"d\s*&\s*o", r"errors", r"directors", r"malpractice",
             r"endorsement", r"exclusion", r"contra", r"beneficial fire", r"exxonmobil"]

def detect_mode(q: str) -> str:
    ql = q.lower()
    return "professional" if any(re.search(t, ql) for t in PRO_TERMS) else "consumer"

def get_embedder():
    global embedder
    if embedder is None:
        embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return embedder

def embed_text(text: str):
    if not index: return [0.0]*384
    return list(get_embedder().embed([text]))[0].tolist()

def search_knowledge(query: str, mode: str):
    if not index: return []
    try:
        vector = embed_text(query)
        top_k = 8 if mode == "professional" else 4
        res = index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [f"{m.metadata.get('title','Source')} — {m.metadata.get('domain','')}" for m in res.matches]
    except: return []

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "TheInsureMaster", "version": "v6.0.2.2", "status": "running"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json() or {}
    question = (data.get("question") or data.get("q") or "").strip()
    jurisdiction = data.get("jurisdiction") or data.get("region") or "US"

    # --- ACCEPT MODE FROM ORIGINAL CHATBOT ---
    mode_input = (data.get("mode") or data.get("type") or data.get("audience") or "").lower()
    if mode_input in ["professional", "pro", "expert", "1"]:
        mode = "professional"
    elif mode_input in ["consumer", "personal", "0"]:
        mode = "consumer"
    else:
        mode = detect_mode(question) if question else "consumer"

    # --- HELLO HANDLER (works with no selection) ---
    if not question or question.lower() in ["hello", "hi", "hey", "hello tim", "hi tim"]:
        greeting = "Hello! I'm TIM v6.0.2.2. I have two modes:\n\n• CONSUMER — plain English for personal insurance\n• PROFESSIONAL — detailed analysis with caselaw\nPick a mode above, then ask your question."
        return jsonify({
            "answer": greeting,
            "response": greeting, # for old chatbot compatibility
            "mode": mode,
            "jurisdiction": jurisdiction,
            "version": "v6.0.2.2"
        })

    if not groq_client:
        return jsonify({"error": "Groq not configured", "response": "Backend not configured"}), 500

    sources = search_knowledge(question, mode)
    context = "\n".join(sources)

    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": f"Question: {question}\nJurisdiction: {jurisdiction}\nSources: {context}"}
        ],
        temperature=0.25 if mode == "consumer" else 0.12,
        max_tokens=900
    )

    answer = completion.choices[0].message.content

    # Return BOTH fields for compatibility
    return jsonify({
        "answer": answer,
        "response": answer,
        "mode": mode,
        "jurisdiction": jurisdiction,
        "sources_used": sources,
        "version": "v6.0.2.2"
    })

@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
