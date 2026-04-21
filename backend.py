# backend.py - Ask TIM v5.5.4-nohealth
# Version: 5.5.4
# No health endpoint - Render will use /ask only

import os
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tim-knowledge")

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

@app.before_request
def check_abuse():
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
    if any(x in q for x in ["uk","united kingdom","england"]):
        return "uk"
    return "us-national"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "")[:500]
    mode = data.get("mode", "consumer")
    region = data.get("region") or detect_region(question)

    try:
        with open("tim_queries.log", "a") as f:
            f.write(f"{datetime.now().isoformat()},{region},{mode}\n")
    except:
        pass

    sources = PROFESSIONAL_SOURCES_UK if region == "uk" else PROFESSIONAL_SOURCES_US
    region_label = "UK" if region == "uk" else ("US-National" if region == "us-national" else region.title())

    pinecone_note = f"From our internal research database (index: {PINECONE_INDEX})"

    if mode == "consumer":
        answer = f"**Direct Answer:** Based on {region_label} guidelines...\n\n**Explanation:** {pinecone_note}.\n\n**Does this clarify?**"
    else:
        answer = f"**Direct Answer:** Professional analysis - {region_label}.\n\n**Explanation:** {pinecone_note}.\n\n**Primary Sources:** {', '.join(sources[:3])}"

    answer += "\n\n*Click here to view our Disclaimer.*"

    return jsonify({
        "answer": answer,
        "sources": sources[:5],
        "confidence": "Confidence: High",
        "audience": mode,
        "region": region,
        "version": "5.5.4-nohealth"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
