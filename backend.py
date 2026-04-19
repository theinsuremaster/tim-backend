import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tim-docs")
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_web(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(f"https://html.duckduckgo.com/html/?q={query}", headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        texts = []
        for result in soup.select('.result__snippet')[:5]:
            texts.append(result.get_text(strip=True))
        return " ".join(texts)
    except:
        return ""

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    
    # 1. Search your documents
    vector = model.encode(question).tolist()
    results = index.query(vector=vector, top_k=3, include_metadata=True)
    
    context = ""
    for match in results.matches:
        if match.score > 0.72:
            context += match.metadata.get('text', '')[:600] + " "
    
    # 2. Search web if needed
    web_context = ""
    if not context or len(context) < 200:
        web_context = search_web(question)
    
    # 3. Build prompt like Meta AI
    full_context = (context + " " + web_context).strip()[:3000]
    
    if not full_context:
        answer = "I don't have enough information to answer that yet. Try asking about insurance, risk management, and finance."
    else:
        # Simple synthesis - use context to answer naturally
        # For a more advanced version, you could call an LLM here
        # For now, we return the most relevant passages cleaned up
        answer = full_context
        # Clean it up to read like Meta AI
        sentences = answer.split('. ')
        answer = '. '.join(sentences[:6])  # Keep it concise
        if not answer.endswith('.'):
            answer += '.'
    
    return jsonify({"answer": answer})

@app.route('/')
def home():
    return jsonify({"status": "Ask TIM ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
