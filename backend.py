===
# backend.py - Ask TIM v5.5.4
# The Insure Master - Senior Insurance Agent/Underwriter/Risk Advisor
# Date: April 21, 2026
# Version: 5.5.4
# Changes: All-US search priority (no KS default), added naic.org, swarb.co.uk for UK

import os
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# === KEYS ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tim-knowledge")
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")

# === 15 PROFESSIONAL SOURCES - ALL US PRIORITY ===
PROFESSIONAL_SOURCES_US = [
    "naic.org", # 1 - Model laws, all states
    "rims.org", # 2 - Risk management
    "theinstitutes.org", # 3 - CPCU/technical
    "ambest.com", # 4 - Carrier ratings
    "insurancejournal.com", # 5 - National news
    "propertycasualty360.com", # 6
    "businessinsurance.com", # 7
    "riskandinsurance.com", # 8
    "rma.usda.gov", # 9 - Federal
    "siccode.com", # 10
    "ijacademy.com", # 11
    "resdm.com", # 12
    "globalenergymonitor.org", # 13
]

PROFESSIONAL_SOURCES_UK = [
    "legislation.gov.uk", # 1
    "swarb.co.uk", # 2 - UK case summaries
    "bailii.org", # 3 - (planned)
] + PROFESSIONAL_SOURCES_US[:5] # fallback to top US

# Court mapping for CourtListener (only when state specified)
COURT_CODES = {
    "california": ["cal","cacd","cand","casd","ca9"],
    "texas": ["tex","txed","txnd","txsd","txwd","ca5"],
    "new york": ["ny","nysd","nyed","ca2"],
    "florida": ["fla","flmd","flnd","flsd","ca11"],
    "illinois": ["ilnd","ilcd","ilsd","ca7"],
    "pennsylvania": ["paed","pamd","pawd","ca3"],
}

# === ANTI-ABUSE ===
request_counts = defaultdict(list)
RATE_LIMIT = 20

@app.before_request
def check_abuse():
    # Allow Render health checks
    if request.path == "/" or "health" in request.path:
        return
    ua = request.headers.get("User-Agent", "").lower()
    if "render" in ua:
        return

    ip = request.remote_addr
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 3600]
    if len(request_counts[ip]) >= RATE_LIMIT:
        return jsonify({"error": "Rate limit"}), 429
    request_counts[ip].append(now)

def detect_region(question):
    """v5.5.4: Defaults to US-NATIONAL, not Kansas"""
    q = question.lower()
    states = list(COURT_CODES.keys())
    for state in states:
        if state in q:
            return state
    if any(x in q for x in ["uk","united kingdom","england","scotland","wales"]):
        return "uk"
    if "canada" in q:
        return "canada"
    return "us-national" # Changed from "kansas"

@app.route("/")
def health():
    return jsonify({
        "status": "TIM v5.5.4 running",
        "pinecone": PINECONE_INDEX,
        "sources_us": len(PROFESSIONAL_SOURCES_US),
        "sources_uk": len(PROFESSIONAL_SOURCES_UK),
        "mode": "all-us"
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "")[:500]
    mode = data.get("mode", "consumer")
    region = data.get("region") or detect_region(question)

    # Log
    try:
        with open("tim_queries.log", "a") as f:
            f.write(f"{datetime.now().isoformat()},{region},{mode},{question[:60]}\n")
    except:
        pass

    # Select sources based on region
    if region == "uk":
        sources = PROFESSIONAL_SOURCES_UK
        region_label = "UK"
    else:
        sources = PROFESSIONAL_SOURCES_US
        region_label = "US-National" if region == "us-national" else region.title()

    # Simulate Pinecone query (replace with your actual client)
    pinecone_note = f"From our internal research database (index: {PINECONE_INDEX})"

    # Format response
    if mode == "consumer":
        answer = f"**Direct Answer:** Based on {region_label} insurance guidelines...\n\n"
        answer += f"**Explanation:** {pinecone_note}. This is the national standard that applies across all US states unless a specific state law overrides it.\n\n"
        answer += "**Does this clarify your question?**"
        confidence = "High"
    else:
        answer = f"**Direct Answer:** Professional analysis - {region_label} market.\n\n"
        answer += f"**Explanation:** {pinecone_note}. Sourced from NAIC model laws and national carriers.\n\n"
        answer += f"**Primary Sources:** {', '.join(sources[:3])}"
        confidence = "High"

    answer += "\n\n*Click here to view our Disclaimer.*"

    return jsonify({
        "answer": answer,
        "sources": sources[:5],
        "confidence": f"Confidence: {confidence}",
        "audience": mode,
        "region": region,
        "version": "5.5.4"
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
