# backend.py - TheInsureMaster v6.0.2.1
# Professional vs Consumer with distinct tone, language, and source depth
# Render-ready with root route

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

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    print(f"Init warning: {e}")
    groq_client = None

# --- DISTINCT SYSTEM PROMPTS WITH TONE ---

SYSTEM_PROMPTS = {
    "consumer": """You are TheInsureMaster v6.0.2.1 — CONSUMER MODE.

VOICE: Warm, direct, Kansas-neighbor tone. Like explaining to Kamran in Overland Park over coffee. No jargon. Use "you" and "your". Short sentences. Plain English.

DEPTH: 180-220 words. Give ONE clear recommendation, then why. Use real numbers ($387/yr, $6,500 ACV). No caselaw. No Latin.

SOURCES: Use only iii.org, bankrate.com, investopedia.com, naic.org. Cite with superscripts ¹²³.

STRUCTURE:
1. Direct answer in first sentence ("Yes — drop it.")
2. Explain in 2 paragraphs with numbers
3. Action step
4. End with: "Want the [specific] calculation?"

Close with:
Sources
- Title — domain
- Title — domain

Consult licensed agent. Disclaimer: https://theinsuremaster.com/disclaimer
Confidence: 0.xx | v6.0.2.1""",

    "professional": """You are TheInsureMaster v6.0.2.1 — PROFESSIONAL MODE.

VOICE: Precise, analytical, coverage-counsel tone. Assume reader is broker, risk manager, or attorney. Use terms: manuscript endorsement, extrinsic evidence, contra proferentem, Side A DIC. No fluff.

DEPTH: 200-240 words. Integrate caselaw into analysis, not as list. Explain holdings and application. Cite at least 2 cases. Discuss jurisdictional splits (CA vs TX).

SOURCES: Use irmi.com, wiley.law, pillsburylaw.com, naic.org, lloyds.com, ambest.com, legislation.gov.uk. Mark paywalls. Cite with superscripts ¹²³⁴.

STRUCTURE:
1. Direct legal conclusion
2. Primary authority with case analysis (*Travelers v. TechAdvisors* held...)
3. Secondary doctrines (*Beneficial Fire* requires... *ExxonMobil* narrows...)
4. Market practice (Pillsbury reports 78%...)
5. Recommendation with specific language

Close with:
Sources
- Title — domain (subscription required)
- Title — domain

Caselaw
1. Full citation
2. Full citation

Consult broker/counsel. Disclaimer: https://theinsuremaster.com/disclaimer
Confidence: 0.xx | v6.0.2.1"""
}

# --- MODE DETECTION (enhanced) ---
PRO_TERMS = [
    r"e\s*&\s*o", r"d\s*&\s*o", r"errors and omissions", r"directors",
    r"malpractice", r"coverage", r"endorsement", r"exclusion", r"manuscript",
    r"contra proferentem", r"extrinsic", r"beneficial fire", r"exxonmobil",
    r"re digitech", r"side a", r"wrongful trading", r"s\.172", r"pool re",
    r"reinsurance", r"wiley", r"irmi", r"pillsbury", r"lloyd'?s", r"ambest"
]

def detect_mode(q: str) -> str:
    ql = q.lower()
    if any(re.search(t, ql) for t in PRO_TERMS):
        return "professional"
    if len(ql.split()) > 18 and any(w in ql for w in ["policy", "cover", "liability", "insurer", "claim"]):
        return "professional"
    return "consumer"

def embed_text(text: str):
    return list(embedder.embed([text]))[0].tolist()

def search_knowledge(query: str, mode: str):
    try:
        vector = embed_text(query)
        top_k = 8 if mode == "professional" else 4
        res = index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [f"{m.metadata.get('title','Source')} — {m.metadata.get('domain','')}" for m in res.matches]
    except:
        return []

# --- ROUTES ---
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "TheInsureMaster",
        "version": "v6.0.2.1",
        "modes": {"consumer": "warm, plain English", "professional": "precise, caselaw-integrated"},
        "endpoints": ["/health", "/ask"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v6.0.2.1"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "")
    jurisdiction = data.get("jurisdiction", "US")
    mode = data.get("mode") or detect_mode(question)

    if not groq_client:
        return jsonify({"error": "Groq not configured"}), 500

    sources = search_knowledge(question, mode)
    context = "\n".join(sources)

    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": f"Question: {question}\nJurisdiction: {jurisdiction}\nContext sources: {context}"}
        ],
        temperature=0.25 if mode == "consumer" else 0.12,
        max_tokens=1000,
        top_p=0.9
    )

    answer = completion.choices[0].message.content

    return jsonify({
        "answer": answer,
        "mode": mode,
        "tone": "warm-plain" if mode == "consumer" else "precise-analytical",
        "jurisdiction": jurisdiction,
        "sources_used": sources,
        "version": "v6.0.2.1"
    })

@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
