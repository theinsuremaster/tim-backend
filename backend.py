import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load models once at startup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tim-docs")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def search_web(query):
    try:
        r = requests.get(f"https://html.duckduckgo.com/html/?q={query}",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')
        snippets = [s.get_text(strip=True) for s in soup.select('.result__snippet')[:5]]
        return " ".join(snippets)
    except:
        return ""

def clean_answer(text, question):
    if len(text) < 100:
        return text
    try:
        # Summarize to Meta AI style - concise and natural
        summary = summarizer(text[:2000], max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except:
        # Fallback if model fails
        sentences = text.split('. ')
        return '. '.join(sentences[:4]) + '.'

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')

    # 1. Get context from your docs
    vector = embed_model.encode(question).tolist()
    results = index.query(vector=vector, top_k=3, include_metadata=True)

    context = ""
    for m in results.matches:
        if m.score > 0.70:
            context += m.metadata.get('text', '') + " "

    # 2. Add web if needed
    if len(context) < 300:
        context += " " + search_web(question)

    if not context.strip():
        return jsonify({"answer": "I don't have information on that yet. Ask me about contracts, insurance, or legal topics."})

    # 3. Create Meta AI style answer
    answer = clean_answer(context, question)

    return jsonify({"answer": answer})

@app.route('/')
def home():
    return jsonify({"status": "Ask TIM with summarizer ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
