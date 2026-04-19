import os
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# --- Setup ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tim-docs")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Priority insurance/risk sites
PRIORITY_SITES = [
    "irmi.com",
    "lexisnexis.com",
    "thomsonreuters.com",
    "iii.org",
    "naic.org",
    "theInstitutes.org",
    "rims.org",
    "ambest.com"
]

DISCLAIMER = "\n\n---\n*This is AI-generated for educational purposes only. Do not rely solely on this information. Consult your licensed insurance agent, risk manager, or financial advisor before making decisions.*"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:4000]

def search_pinecone(question):
    vector = embed_model.encode(question).tolist()
    results = index.query(vector=vector, top_k=5, include_metadata=True)
    context = []
    for m in results.matches:
        if m.score > 0.68:
            txt = m.metadata.get('text', '')
            context.append(txt)
    return " ".join(context)

def search_web_priority(question):
    """Search web with priority to insurance sites"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        # Build query prioritizing insurance sites
        site_query = " OR ".join([f"site:{s}" for s in PRIORITY_SITES])
        query = f"{question} ({site_query})"
        
        r = requests.get(f"https://html.duckduckgo.com/html/?q={query}", 
                        headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        results = []
        for result in soup.select('.result')[:6]:
            title = result.select_one('.result__title')
            snippet = result.select_one('.result__snippet')
            link = result.select_one('.result__url')
            
            if snippet:
                text = snippet.get_text(strip=True)
                source = link.get_text(strip=True) if link else "web"
                results.append(f"{text} [Source: {source}]")
        
        # Fallback to general web if no priority results
        if not results:
            r2 = requests.get(f"https://html.duckduckgo.com/html/?q={question}", 
                            headers=headers, timeout=8)
            soup2 = BeautifulSoup(r2.text, 'html.parser')
            results = [s.get_text(strip=True) for s in soup2.select('.result__snippet')[:4]]
        
        return " ".join(results)
    except Exception as e:
        return ""

def auto_learn(question, web_content):
    """Save web findings to Pinecone for future learning"""
    try:
        if len(web_content) > 100:
            vector = embed_model.encode(question).tolist()
            # Use timestamp as ID to avoid duplicates
            doc_id = f"web_{int(datetime.now().timestamp())}"
            index.upsert(vectors=[{
                "id": doc_id,
                "values": vector,
                "metadata": {
                    "text": web_content[:1000],
                    "source": "web_auto_learn",
                    "topic": "insurance",
                    "date": datetime.now().isoformat()
                }
            }])
    except:
        pass  # Don't break if learning fails

def format_meta_ai_style(question, context):
    """Format like Meta AI - clean, conversational, with structure"""
    if not context:
        return "I don't have enough information on that topic yet."
    
    # Clean and structure
    sentences = context.split('. ')
    main_points = sentences[:6]
    
    # Meta AI style formatting
    answer = f"**{question}**\n\n"
    
    # Add key points with bullets if it's a complex topic
    if len(main_points) > 3 and any(word in question.lower() for word in ['what', 'how', 'explain', 'difference']):
        answer += "Here's what you should know:\n\n"
        for i, point in enumerate(main_points[:4], 1):
            if len(point) > 20:
                answer += f"• {point.strip()}.\n"
    else:
        answer += ". ".join(main_points) + "."
    
    return answer

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"answer": "Please ask a question about insurance, risk management, or finance."})
    
    # 1. Search your documents
    pinecone_context = search_pinecone(question)
    
    # 2. Search priority web sources
    web_context = ""
    if len(pinecone_context) < 300:
        web_context = search_web_priority(question)
        # Auto-learn for next time
        if web_context:
            auto_learn(question, web_context)
    
    # 3. Combine contexts
    full_context = clean_text(pinecone_context + " " + web_context)
    
    # 4. Format like Meta AI
    answer = format_meta_ai_style(question, full_context)
    
    # 5. Add disclaimer
    answer += DISCLAIMER
    
    return jsonify({
        "answer": answer,
        "sources_checked": ["your_docs"] + (["IRMI, LexisNexis, Thomson Reuters, industry sites"] if web_context else [])
    })

@app.route('/health')
def health():
    return jsonify({"status": "Ask TIM Insurance AI ready", "version": "2.0"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
