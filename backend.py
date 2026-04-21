"""
Ask TIM v5.0 - Complete Backend
- Location-aware (Americas/Europe/Asia-Pacific)
- Website-first search (theinsuremaster.com)
- Pinecone RAG with auto-learning
- 4-part output order
- Single-line markdown references
"""

import os
import re
import json
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from docx import Document
import io

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ask-tim")
PORT = int(os.getenv("PORT", 5000))

app = Flask(__name__)
CORS(app)

# Init clients
client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(PINECONE_INDEX) if pc else None
embedder = TextEmbedding('BAAI/bge-small-en-v1.5')

# --- HELPERS ---

def get_user_region():
    country = request.headers.get('CF-IPCountry') or request.headers.get('X-Country', 'US')
    country = country.upper()
    if country in ['US','CA','MX','BR','AR']: return 'Americas'
    if country in ['GB','DE','FR','IT','ES','NL','SE','CH']: return 'Europe'
    if country in ['IN','JP','CN','SG','AU','HK','KR']: return 'Asia-Pacific'
    return 'Americas'

def search_insuremaster(query):
    """Search theinsuremaster.com first"""
    try:
        url = f"https://theinsuremaster.com/?s={requests.utils.quote(query)}"
        r = requests.get(url, timeout=6, headers={"User-Agent":"AskTIM/5.0"})
        soup = BeautifulSoup(r.text, 'lxml')
        results = []
        for item in soup.select('article')[:3]:
            a = item.select_one('h2 a, h3 a')
            if not a: continue
            title = a.get_text(strip=True)
            link = a['href']
            price = ""
            # check if product
            if '/product/' in link or 'shop' in link:
                try:
                    p = requests.get(link, timeout=4)
                    ps = BeautifulSoup(p.text, 'lxml')
                    pr = ps.select_one('.price .amount, .woocommerce-Price-amount')
                    if pr: price = f" - {pr.get_text(strip=True)}"
                except: pass
            results.append({"title": title, "url": link, "price": price, "source": "insuremaster"})
        return results
    except Exception as e:
        print("Site search error:", e)
        return []

def search_pinecone(query, region):
    if not index: return []
    try:
        q_emb = list(embedder.embed([query]))[0].tolist()
        res = index.query(vector=q_emb, top_k=3, include_metadata=True,
                          filter={"region": {"$in": [region, "global"]}})
        out = []
        for m in res.get('matches', []):
            md = m['metadata']
            out.append({"title": md.get('title','Document'), "url": md.get('url',''), "source": "pinecone"})
        return out
    except Exception as e:
        print("Pinecone error:", e)
        return []

def filter_bad_sources(results):
    bad = ['weblio.jp','baidu.com','.cn','.jp','wikipedia.org','zhihu.com']
    clean = []
    for r in results:
        url = r.get('url','').lower()
        if any(b in url for b in bad): continue
        clean.append(r)
    return clean

def build_references(website_results, pinecone_results):
    refs = []
    # website first
    for r in website_results[:2]:
        title = r['title'] + r.get('price','')
        refs.append(f"- [{title}]({r['url']})")
    for r in pinecone_results[:2]:
        if r['url']:
            refs.append(f"- [{r['title']}]({r['url']})")
    if not refs:
        refs.append("- [The InsureMaster Knowledge Base](https://theinsuremaster.com)")
    return "\n".join(refs)

TIM_PROMPT = """You are Ask TIM, the AI assistant for theinsuremaster.com.

USER REGION: {region}

You MUST answer in this EXACT 4-part format:

1. DIRECT ANSWER:
[1-2 sentences, straight to point]

2. EXPLANATION:
[3-4 sentences. Include relevant case law or regulation from {region} ONLY. Americas=US/Canada, Europe=UK/EU, Asia-Pacific=local. Never cite other regions.]

3. REFERENCES:
{references}

4. CONFIDENCE: [High/Medium/Low]

RULES:
- Use information from theinsuremaster.com first.
- Never use Japanese dictionaries, weblio.jp, baidu, or non-English blogs.
- Format references as single-line markdown links, never full raw URLs.
- If a product/checklist is found, mention price in reference title.
- Be concise and professional.
"""

@app.route('/health')
def health():
    return jsonify({"status":"ok","version":"5.0"})

@app.route('/ask', methods=['GET','POST'])
def ask():
    q = request.args.get('q') or (request.json or {}).get('q', '')
    if not q: return jsonify({"error":"No query"}),400
    
    region = get_user_region()
    
    # 1. Search website first
    site_results = search_insuremaster(q)
    site_results = filter_bad_sources(site_results)
    
    # 2. Search Pinecone
    pc_results = search_pinecone(q, region)
    
    # 3. Build references string
    refs_md = build_references(site_results, pc_results)
    
    # 4. Build context
    context = f"Website matches: {json.dumps(site_results[:2])}"
    
    # 5. Call Groq
    try:
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role":"system","content": TIM_PROMPT.format(region=region, references=refs_md)},
                {"role":"user","content": f"Query: {q}\nContext: {context}"}
            ],
            temperature=0.1,
            max_tokens=700
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"1. DIRECT ANSWER:\nService temporarily unavailable.\n\n2. EXPLANATION:\nError: {str(e)}\n\n3. REFERENCES:\n{refs_md}\n\n4. CONFIDENCE: Low"
    
    # Log query for future repository
    try:
        with open("/tmp/queries.log","a") as f:
            f.write(f"{datetime.utcnow().isoformat()},{region},{q}\n")
    except: pass
    
    return jsonify({
        "query": q,
        "region": region,
        "answer": answer,
        "website_results": site_results
    })

# --- Document analysis endpoint (kept from v4) ---
def extract_text(file):
    name = secure_filename(file.filename).lower()
    data = file.read()
    if name.endswith('.pdf'):
        doc = fitz.open(stream=data, filetype='pdf')
        return "\n".join(p.get_text() for p in doc)[:20000]
    if name.endswith('.docx'):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)[:20000]
    if name.endswith('.txt'):
        return data.decode('utf-8','ignore')[:20000]
    return ""

@app.route('/analyze-document', methods=['POST'])
def analyze_document():
    if 'file' not in request.files: return jsonify({"error":"No file"}),400
    f = request.files['file']
    text = extract_text(f)
    region = get_user_region()
    prompt = f"Analyze this document for {region} insurance purposes. Follow the 4-part format."
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role":"system","content":TIM_PROMPT.format(region=region, references="[Document Analysis]")},
                  {"role":"user","content": prompt + "\n\n" + text[:15000]}],
        temperature=0.1
    )
    return jsonify({"analysis": resp.choices[0].message.content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
