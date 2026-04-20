import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Initialize clients
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# CHANGE: Using your index name
INDEX_NAME = "tim-knowledge"
index = pc.Index(INDEX_NAME)

# Lightweight embedder (replaces sentence-transformers)
embed = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

SYSTEM = """You are 'Ask TIM,' the authoritative AI assistant for The InsureMaster.
Core Knowledge: Insurance, risk management, and financial services.
Research Mandate: Synthesize from: 1) Internal knowledge base (Pinecone), 2) IRMI, NAIC, III, Investopedia, 3) Your base knowledge.
Output Style: Write like Meta AI — conversational, clear, with headings and bullets. Cite sources as [Source].
Audience: Adapt for 'professional' (technical) vs 'consumer' (simple).
Disclaimer: End every answer with: '---\n*This information is for educational purposes. Consult a licensed professional for advice specific to your situation.*'
"""

def get_internal(question):
    """Search Pinecone tim-knowledge index"""
    try:
        # FastEmbed returns generator, get first vector
        vec = list(embed.embed(question))[0].tolist()

        results = index.query(
            vector=vec,
            top_k=3,
            include_metadata=True
        )

        texts = []
        for match in results.matches:
            text = match.metadata.get('text', '')[:400]
            source = match.metadata.get('source', 'TIM Knowledge')
            page = match.metadata.get('page', '')
            citation = f"{source} p.{page}" if page else source
            texts.append(f"[{citation}] {text}")

        return "\n\n".join(texts) if texts else "No internal matches found."
    except Exception as e:
        return f"Internal knowledge base temporarily unavailable."

def get_web(question):
    """Lightweight web search from authoritative sources"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        # Limit to insurance authorities
        sites = "site:irmi.com OR site:naic.org OR site:iii.org OR site:investopedia.com"
        url = f"https://html.duckduckgo.com/html/?q={question} {sites}"

        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')

        results = []
        for result in soup.select('.result')[:3]:
            snippet = result.select_one('.result__snippet')
            if snippet:
                text = snippet.get_text(strip=True)[:280]
                # Detect source
                html = str(result).lower()
                if 'irmi' in html:
                    source = 'IRMI'
                elif 'naic' in html:
                    source = 'NAIC'
                elif 'iii.org' in html:
                    source = 'III'
                else:
                    source = 'WEB'
                results.append(f"[{source}] {text}")

        return "\n\n".join(results) if results else "No recent web sources found."
    except Exception as e:
        return "Web search temporarily unavailable."

@app.route("/")
def home():
    return jsonify({
        "status": "Ask TIM operational",
        "index": INDEX_NAME,
        "mode": "pinecone+web+groq"
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    audience = data.get("audience", "consumer")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Get both sources
    internal = get_internal(question)
    web = get_web(question)

    # Audience instruction
    tone = "Explain in plain language, avoid jargon, use examples." if audience == "consumer" else "Use precise insurance terminology, assume CPCU-level knowledge."

    user_prompt = f"""Question: {question}

AUDIENCE: {audience.upper()} - {tone}

INTERNAL KNOWLEDGE BASE (tim-knowledge):
{internal}

LIVE WEB RESEARCH:
{web}

INSTRUCTIONS:
1. Synthesize a comprehensive answer using both sources
2. Prioritize internal knowledge, supplement with web
3. Format with clear headings and bullet points
4. Cite sources inline like [IRMI] or [TIM Knowledge p.23]
5. Adapt complexity for {audience} audience"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1800
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    return jsonify({
        "answer": answer,
        "sources": {
            "internal": "tim-knowledge" in internal.lower() or "TIM" in internal,
            "web": "IRMI" in web or "NAIC" in web
        },
        "audience": audience
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
