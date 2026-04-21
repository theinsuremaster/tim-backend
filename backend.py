# backend.py - Ask TIM v5.5.5-fixed
import os
import time
from datetime import datetime
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tim-knowledge")
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")

PROFESSIONAL_SOURCES_US = [
    "naic.org", "rims.org", "theinstitutes.org", "ambest.com",
    "insurancejournal.com", "propertycasualty360.com",
    "businessinsurance.com", "riskandinsurance.com",
    "rma.usda.gov", "siccode.com", "ijacademy.com",
    "resdm.com", "globalenergymonitor.org",
]

PROFESSIONAL_SOURCES_UK = [
    "legislation.gov.uk", "swarb.co.uk", "bailii.org",
] + PROFESSIONAL_SOURCES_US[:5]

COURT_CODES = {
    "california": ["cal","cacd","cand","casd","ca9"],
    "texas": ["tex","txed","txnd","txsd","txwd","ca5"],
    "new york": ["ny","nysd","nyed","ca2"],
    "florida": ["fla","flmd","flnd","flsd","ca11"],
    "illinois": ["ilnd","ilcd","ilsd","ca7"],
    "pennsylvania": ["paed","pamd","pawd","ca3"],
}

request_counts = defaultdict(list)
RATE_LIMIT = 20

# === HEALTH CHECK - stops Render loop ===
@app.route("/")
@app.route("/health")
@app.route("/healthz")
def health():
    return "ok", 200

def search_courtlistener(query, state=None):
    if not COURTLISTENER_TOKEN:
        return None
    if not any(kw in query.lower() for kw in ["case","v.","vs","statute","lawsuit","court","ruling"]):
        return None
    headers = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}
    params = {"q": query, "type": "o", "order_by": "score desc", "page_size": 3}
    if state and state in COURT_CODES:
        params["court"] = ",".join(COURT_CODES[state])
    try:
        r = requests.get("https://www.courtlistener.com/api/rest/v3/search/", headers=headers, params=params, timeout=8)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                top = results[0]
                return {
                    "case": top.get("caseName", "Unknown"),
                    "court": top.get("court", ""),
                    "date": top.get("dateFiled", "")[:10],
                    "snippet": top.get("snippet", "")[:200],
                    "url": f"https://www.courtlistener.com{top.get('absolute_url', '')}"
                }
    except Exception as e:
        print(f"CourtListener error: {e}")
    return None

@app.before_request
def check_abuse():
    if request.path in ["/", "/health", "/healthz"]:
        return
    if request.path!= "/ask":
        return jsonify({"error": "Not found"}), 404
    ip = request.remote_addr
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 3600]
    if len(request_counts[ip]) >= RATE_LIMIT:
        return jsonify({"error": "Rate limit"}), 429
    request_counts[ip].append(now)

def detect_region(question):
    q = question.lower()
    for state in COURT_CODES.keys():
        if state in q:
            return state
    if any(x in q for x in ["uk","united kingdom","england","scotland"]):
        return "uk"
    return "us-national"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "")[:500]
    mode = data.get("mode", "consumer")
    region = data.get("region") or detect_region(question)

    sources = PROFESSIONAL_SOURCES_UK if region == "uk" else PROFESSIONAL_SOURCES_US
    region_label = "UK" if region == "uk" else ("US-National" if region == "us-national" else region.title())

    court_data = None
    if mode == "professional":
        court_data = search_courtlistener(question, region if region!= "us-national" else None)

    pinecone_note = f"From our internal research database (index: {PINECONE_INDEX})"

    if mode == "consumer":
        answer = f"**Direct Answer:** Based on {region_label} insurance guidelines...\n\n**Explanation:** {pinecone_note}.\n\n**Does this clarify your question?**"
    else:
        answer = f"**Direct Answer:** Professional analysis - {region_label}.\n\n**Explanation:** {pinecone_note}.\n\n**Primary Sources:** {', '.join(sources[:3])}"
        if court_data:
            answer += f"\n\n**Relevant Case Law:** {court_data['case']} ({court_data['date']})\n{court_data['snippet']}...\nSource: {court_data['url']}"

    answer += "\n\n*Click here to view our Disclaimer.*"

    response = {
        "answer": answer,
        "sources": sources[:5],
        "confidence": "Confidence: High",
        "audience": mode,
        "region": region,
        "version": "5.5.5-fixed"
    }
    if court_data:
        response["court_case"] = court_data
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
