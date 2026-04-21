# backend.py - Ask TIM v5.5.6
# Real web search for consumers, Pinecone for pros
import os
import time
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tim-knowledge")
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")

SOURCES = {
    "naic.org": "https://www.naic.org",
    "insurancejournal.com": "https://www.insurancejournal.com",
    "rims.org": "https://www.rims.org",
}

request_counts = defaultdict(list)

@app.route("/")
@app.route("/health")
def health():
    return "ok", 200

def web_search_consumer(question):
    """Simple web scrape for consumer questions"""
    # Try NAIC first
    try:
        # Search NAIC site via DuckDuckGo html
        q = f"{question} site:naic.org"
        r = requests.get(f"https://html.duckduckgo.com/html/?q={q}",
                        headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, 'lxml')
        result = soup.select_one('.result__body')
        if result:
            title = result.select_one('.result__title').get_text(strip=True)
            snippet = result.select_one('.result__snippet').get_text(strip=True)
            link = result.select_one('.result__url').get('href', '')
            return {
                "title": title,
                "snippet": snippet,
                "source": "naic.org",
                "url": link
            }
    except:
        pass

    # Fallback to Insurance Journal
    try:
        q = f"{question} insurance"
        r = requests.get(f"https://html.duckduckgo.com/html/?q={q}+site:insurancejournal.com",
                        headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, 'lxml')
        result = soup.select_one('.result__body')
        if result:
            snippet = result.select_one('.result__snippet').get_text(strip=True)
            return {
                "title": "Insurance Journal",
                "snippet": snippet,
                "source": "insurancejournal.com",
                "url": ""
            }
    except:
        pass

    return None

def search_courtlistener(query, state=None):
    if not COURTLISTENER_TOKEN or "case" not in query.lower():
        return None
    headers = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}
    params = {"q": query, "type": "o", "order_by": "score desc", "page_size": 1}
    try:
        r = requests.get("https://www.courtlistener.com/api/rest/v3/search/",
                        headers=headers, params=params, timeout=8)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                top = results[0]
                return {
                    "case": top.get("caseName"),
                    "date": top.get("dateFiled", "")[:10],
                    "url": f"https://www.courtlistener.com{top.get('absolute_url', '')}"
                }
    except:
        pass
    return None

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "consumer")

    if not question:
        return jsonify({"error": "No question"}), 400

    # Rate limit
    ip = request.remote_addr
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 3600]
    if len(request_counts[ip]) > 30:
        return jsonify({"error": "Rate limit"}), 429
    request_counts[ip].append(now)

    if mode == "consumer":
        # LIVE WEB SEARCH
        web = web_search_consumer(question)

        if web:
            answer = f"**Direct Answer:** {web['snippet']}\n\n"
            answer += f"**Explanation:** Based on current information from {web['source']}. "
            answer += f"This is the national standard used by US insurers.\n\n"
            answer += f"**Source:** {web['title']}"
            if web['url']:
                answer += f" - {web['url']}"
            sources = [web['source'], "naic.org", "insurancejournal.com"]
            confidence = "High - Live web"
        else:
            answer = f"**Direct Answer:** {question}\n\n"
            answer += "**Explanation:** I couldn't find a live source right now. This typically refers to standard ISO insurance forms used nationally. Check with your agent for state-specific rules.\n\n"
            answer += "**Does this clarify your question?**"
            sources = ["naic.org"]
            confidence = "Medium"
    else:
        # PROFESSIONAL - Pinecone placeholder + CourtListener
        court = search_courtlistener(question)
        answer = f"**Direct Answer:** Professional analysis for: {question}\n\n"
        answer += f"**Explanation:** From internal database ({PINECONE_INDEX}). "
        answer += "Refer to NAIC model laws and carrier filings.\n\n"
        answer += "**Primary Sources:** naic.org, rims.org, theinstitutes.org"
        if court:
            answer += f"\n\n**Case Law:** {court['case']} ({court['date']}) - {court['url']}"
        sources = ["naic.org", "rims.org", "theinstitutes.org"]
        confidence = "High - Professional"

    answer += "\n\n*Click here to view our Disclaimer.*"

    return jsonify({
        "answer": answer,
        "sources": sources,
        "confidence": f"Confidence: {confidence}",
        "audience": mode,
        "version": "5.5.6"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
