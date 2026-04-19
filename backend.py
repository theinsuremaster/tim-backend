import os, re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tim-docs")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

PRIORITY_SITES = ["irmi.com","lexisnexis.com","thomsonreuters.com","iii.org","naic.org","rims.org","ambest.com"]
DISCLAIMER = "\n\n---\n*AI-generated for educational purposes only. Consult your licensed insurance agent, risk manager, or financial advisor before making decisions.*"

def search_pinecone(q):
    vec = embed_model.encode(q).tolist()
    res = index.query(vector=vec, top_k=5, include_metadata=True)
    return " ".join([m.metadata.get('text','') for m in res.matches if m.score>0.68])

def search_web(q):
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        sites = " OR ".join([f"site:{s}" for s in PRIORITY_SITES])
        r = requests.get(f"https://html.duckduckgo.com/html/?q={q} ({sites})", headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        snippets = [s.get_text(strip=True) for s in soup.select('.result__snippet')[:5]]
        return " ".join(snippets)
    except:
        return ""

def auto_learn(q, content):
    try:
        if len(content)>100:
            vec = embed_model.encode(q).tolist()
            index.upsert(vectors=[{"id":f"web_{int(datetime.now().timestamp())}","values":vec,"metadata":{"text":content[:1000],"source":"web"}}])
    except: pass

@app.route('/ask', methods=['POST'])
def ask():
    q = request.json.get('question','')
    pine = search_pinecone(q)
    web = ""
    if len(pine) < 300:
        web = search_web(q)
        auto_learn(q, web)
    
    context = (pine + " " + web)[:4000]
    sentences = re.split(r'\.\s+', context)
    answer = f"**{q}**\n\n" + "\n".join([f"• {s.strip()}." for s in sentences[:4] if len(s)>20])
    if not answer.strip(): answer = "I don't have enough information yet."
    
    return jsonify({"answer": answer + DISCLAIMER})

@app.route('/')
def home():
    return jsonify({"status":"ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))
