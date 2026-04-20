import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt — exactly as you designed
SYSTEM = """You are 'Ask TIM,' the authoritative AI assistant for The InsureMaster.
Core Knowledge: Your expertise is grounded in insurance, risk management, and financial services.
Research Mandate: When user asks, you MUST synthesize information from: 1) Internal books, 2) IRMI, NAIC, III, and other authoritative web sources, 3) Your base knowledge.
Output Style: Write like Meta AI — conversational, clear, well-formatted with headings and bullets. Cite sources as [Source]. Always paraphrase, never copy verbatim.
Audience: Adapt tone for 'professional' vs 'consumer' as specified.
Disclaimer: End every answer with: '---\n*This information is for educational purposes. Consult a licensed professional for advice specific to your situation.*'
"""

def get_web_results(question):
    """Lightweight DuckDuckGo scraper - no API keys"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        # Search authoritative sources only
        sites = "site:irmi.com OR site:naic.org OR site:iii.org OR site:investopedia.com"
        url = f"https://html.duckduckgo.com/html/?q={question} {sites}"

        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')

        results = []
        for result in soup.select('.result')[:4]:
            title = result.select_one('.result__title')
            snippet = result.select_one('.result__snippet')
            if snippet:
                text = snippet.get_text(strip=True)[:300]
                source = "IRMI" if "irmi" in str(result) else "NAIC" if "naic" in str(result) else "III" if "iii" in str(result) else "WEB"
                results.append(f"[{source}] {text}")
        return "\n\n".join(results) if results else "No recent web sources found."
    except Exception as e:
        return f"Web search temporarily unavailable."

@app.route("/")
def home():
    return jsonify({"status": "Ask TIM operational", "mode": "free-tier"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    audience = data.get("audience", "consumer")

    # 1. Get live web data
    web_data = get_web_results(question)

    # 2. Build prompt for Groq
    audience_instruction = "Explain simply, avoid jargon." if audience == "consumer" else "Use technical insurance terminology, assume professional knowledge."

    user_prompt = f"""Question: {question}
Audience: {audience} - {audience_instruction}

LIVE WEB RESEARCH:
{web_data}

Synthesize a comprehensive answer using the web research above. Format with headings and bullets like Meta AI. Cite sources inline as [IRMI], [NAIC], etc."""

    # 3. Get Groq response
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    return jsonify({
        "answer": answer,
        "sources_used": ["Web Search", "Groq Llama 3.3"],
        "audience": audience
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
