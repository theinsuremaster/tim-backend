import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tim-docs")
model = SentenceTransformer('all-MiniLM-L6-v2')

BRAVE_KEY = os.getenv("BRAVE_API_KEY")

def search_web(query):
    try:
        # Try Brave first if key exists
        if BRAVE_KEY:
            headers = {"X-Subscription-Token": BRAVE_KEY}
            r = requests.get(f"https://api.search.brave.com/res/v1/web/search?q={query}&count=3", headers=headers, timeout=10)
            data = r.json()
            results = [f"{item['title']}: {item['description']}" for item in data.get('web',{}).get('results',[])[:3]]
            return "\n".join(results), "Brave Search"
    except:
        pass

    # Fallback: free DuckDuckGo html
    try:
        r = requests.get(f"https://html.duckduckgo.com/html/?q={query}", headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []
        for result in soup.select('.result__body')[:3]:
            title = result.select_one('.result__title').get_text(strip=True)
            snippet = result.select_one('.result__snippet').get_text(strip=True)
            results.append(f"{title}: {snippet}")
        return "\n".join(results), "Web Search"
    except Exception as e:
        return f"Web search failed: {e}", "Error"

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')

    # 1. Search Pinecone
    vector = model.encode(question).tolist()
    results = index.query(vector=vector, top_k=3, include_metadata=True)

    docs_text = ""
    best_score = 0
    if results.matches:
        best_score = results.matches[0].score
        for match in results.matches:
            if match.score > 0.7:
                docs_text += match.metadata.get('text', '')[:500] + "\n\n"

    # 2. Decide source
    if docs_text and best_score > 0.75:
        answer = f"[From your documents]\n\n{docs_text[:1500]}"
        source = "docs"
    else:
        web_text, source_name = search_web(question)
        if docs_text:
            answer = f"[From your documents + {source_name}]\n\nYour docs say: {docs_text[:800]}\n\nWeb results: {web_text[:800]}"
        else:
            answer = f"[From {source_name}]\n\n{web_text}"
        source = "web"

    return jsonify({"answer": answer, "source": source})

@app.route('/')
def home():
    return jsonify({"status": "Ask TIM hybrid - ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
