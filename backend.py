import os, re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# --- Keys ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tim-docs")
embed = SentenceTransformer('all-MiniLM-L6-v2')
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

PRIORITY = ["irmi.com","lexisnexis.com","thomsonreuters.com","iii.org","naic.org","rims.org","ambest.com","investopedia.com"]

DISCLAIMER = "\n\n---\n**Disclaimer:** This information is for educational purposes only. It may be incomplete or inaccurate. Always consult a licensed professional or qualified expert before making decisions based on this information."

def get_internal(question):
    vec = embed.encode(question).tolist()
    res = index.query(vector=vec, top_k=5, include_metadata=True)
    texts = [m.metadata.get('text','') for m in res.matches if m.score>0.68]
    return " ".join(texts)[:3000]

def get_external(question):
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        sites = " OR ".join([f"site:{s}" for s in PRIORITY])
        url = f"https://html.duckduckgo.com/html/?q={question} ({sites})"
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []
        for item in soup.select('.result')[:5]:
            title = item.select_one('.result__title')
            snippet = item.select_one('.result__snippet')
            link = item.select_one('.result__url')
            if snippet and link:
                source = link.get_text(strip=True).split('.')[0]
                results.append({
                    "text": snippet.get_text(strip=True),
                    "source": source.upper()
                })
        return results
    except:
        return []

def learn(question, content):
    try:
        if content:
            vec = embed.encode(question).tolist()
            index.upsert([{"id":f"learn_{int(datetime.now().timestamp())}","values":vec,"metadata":{"text":content[:1000],"source":"web"}}])
    except: pass

def synthesize(question, audience, internal, external):
    # Build prompt following system rules
    mode = "PROFESSIONAL" if audience=="professional" else "GENERAL CONSUMER"

    external_text = "\n".join([f"- {e['text']} (SOURCE:{e['source']})" for e in external])
    sources = " ".join([f"🔗 {e['source']}" for e in external[:3]])

    system_prompt = f"""You are The InsureMaster AI Assistant.
Audience: {mode}
Rules:
- Use internal knowledge but NEVER quote it verbatim, never reveal source
- Paraphrase external sources, cite with icons only
- If PROFESSIONAL: use technical terms, frameworks, underwriting logic
- If GENERAL CONSUMER: use plain language, step-by-step, no jargon
- Format with short paragraphs, bullets, or tables
- Never give binding advice"""

    user_prompt = f"""Question: {question}
Internal knowledge (paraphrase only): {internal}
External findings: {external_text}

Provide a clear answer for a {mode.lower()}. End with sources as icons."""

    try:
        resp = groq.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        answer = resp.choices[0].message.content
        if sources:
            answer += f"\n\nSources: {sources}"
        return answer + DISCLAIMER
    except Exception as e:
        return f"Unable to generate response. {DISCLAIMER}"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question','').strip()
    audience = data.get('audience','').lower() # 'professional' or 'consumer'

    if not question:
        return jsonify({"answer":"Please ask a question."})

    # Dual-audience check
    if not audience:
        return jsonify({
            "ask_audience": True,
            "answer": "Are you an insurance/risk/finance professional, or a general consumer?"
        })

    # Get knowledge
    internal = get_internal(question)
    external = get_external(question)

    # Learn
    if external:
        learn(question, " ".join([e['text'] for e in external]))

    # Synthesize with Groq
    answer = synthesize(question, audience, internal, external)

    return jsonify({"answer": answer, "audience": audience})

@app.route('/')
def health():
    return jsonify({"status":"InsureMaster AI ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))
