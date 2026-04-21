"""
Ask TIM v5.2 - Crash-proof for Render
- Uses correct Pinecone index: tim-knowledge
- Lazy init to avoid startup crash
"""

import os
import json
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import traceback

app = Flask(__name__)
CORS(app)

# --- Lazy clients ---
_groq_client = None
_pinecone_index = None
_embedder = None

def get_groq():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key:
                print("WARNING: GROQ_API_KEY not set")
                return None
            _groq_client = Groq(api_key=key)
            print("Groq initialized")
        except Exception as e:
            print("Groq init error:", e)
            traceback.print_exc()
            return None
    return _groq_client

def get_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        try:
            from pinecone import Pinecone
            key = os.getenv("PINECONE_API_KEY")
            if not key:
                print("INFO: PINECONE_API_KEY not set - skipping")
                return None
            pc = Pinecone(api_key=key)
            # CORRECT INDEX NAME
            idx_name = os.getenv("PINECONE_INDEX", "tim-knowledge")
            _pinecone_index = pc.Index(idx_name)
            print(f"Pinecone initialized: {idx_name}")
        except Exception as e:
            print("Pinecone init error:", e)
            traceback.print_exc()
            return None
    return _pinecone_index

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding('BAAI/bge-small-en-v1.5')
        except Exception as e:
            print("Embedder error:", e)
            return None
    return _embedder

def get_user_region():
    country = request.headers.get('CF-IPCountry', 'US').upper()
    if country in ['US','CA','MX','BR','AR']: return 'Americas'
    if country in ['GB','DE','FR','IT','ES','NL','SE','CH']: return 'Europe'
    if country in ['IN','JP','CN','SG','AU','HK','KR']: return 'Asia-Pacific'
    return 'Americas'

def search_insuremaster(query):
    try:
        url = f"https://theinsuremaster.com/?s={requests.utils.quote(query)}"
        r = requests.get(url, timeout=5, headers={"User-Agent":"AskTIM/5.2"})
        soup = BeautifulSoup(r.text, 'lxml')
        results = []
        for item in soup.select('article')[:3]:
            a = item.select_one('h2 a, h3 a')
            if not a: continue
            title = a.get_text(strip=True)
            link = a['href']
            price = ""
            if '/product/' in link:
                try:
                    p = requests.get(link, timeout=3)
                    ps = BeautifulSoup(p.text, 'lxml')
                    pr = ps.select_one('.price .amount, .woocommerce-Price-amount')
                    if pr: price = f" - {pr.get_text(strip=True)}"
                except: pass
            results.append({"title": title, "url": link, "price": price})
        return results
    except Exception as e:
        print("Site search error:", e)
        return []

def search_pinecone(query, region):
    idx = get_pinecone()
    emb = get_embedder()
    if not idx or not emb: return []
    try:
        q_emb = list(emb.embed([query]))[0].tolist()
        res = idx.query(vector=q_emb, top_k=3, include_metadata=True,
                       filter={"region": {"$in": [region.lower(), "global"]}})
        out = []
        for m in res.get('matches', []):
            md = m.get('metadata', {})
            out.append({"title": md.get('title','Knowledge Base'), "url": md.get('url',''), "source": "tim-knowledge"})
        return out
    except Exception as e:
        print("Pinecone query error:", e)
        return []

TIM_PROMPT = """You are Ask TIM for theinsuremaster.com. USER REGION: {region}

Respond in EXACT format:

1. DIRECT ANSWER:
[1-2 sentences]

2. EXPLANATION:
[3-4 sentences with {region} law/cases only]

3. REFERENCES:
{refs}

4. CONFIDENCE: High/Medium/Low
"""

@app.route('/', methods=['GET', 'HEAD'])
def home():
    if request.method == 'HEAD':
        return '', 200
    
    q = request.args.get('q')
    if q:
        return ask()
    
    # Browser gets HTML, API gets JSON
    if 'text/html' in request.headers.get('Accept', ''):
        return f"""<html><body style="font-family:system-ui;padding:40px">
        <h2>✓ Ask TIM v5.3 live</h2>
        <p>Pinecone: tim-knowledge | Groq: {'OK' if get_groq() else 'missing'}</p>
        <form><input name="q" placeholder="Ask..." style="width:300px"><button>Ask</button></form>
        </body></html>"""
    
    return jsonify({"status":"live","version":"5.3","pinecone_index":"tim-knowledge"})

@app.route('/health', methods=['GET'])
def health():
    # keep /health for backwards compatibility
    return home()
    
    region = get_user_region()
    
    # 1. Website first
    site_results = search_insuremaster(q)
    # 2. Pinecone second (tim-knowledge)
    pc_results = search_pinecone(q, region)
    
    # Build refs - single line markdown
    refs = []
    for r in site_results[:2]:
        title = r['title'] + r.get('price','')
        refs.append(f"- [{title}]({r['url']})")
    for r in pc_results[:1]:
        if r['url']:
            refs.append(f"- [{r['title']}]({r['url']})")
    if not refs:
        refs.append("- [The InsureMaster](https://theinsuremaster.com)")
    refs_md = "\n".join(refs)
    
    client = get_groq()
    if not client:
        answer = f"1. DIRECT ANSWER:\nConfiguration needed.\n\n2. EXPLANATION:\nGROQ_API_KEY missing in Render.\n\n3. REFERENCES:\n{refs_md}\n\n4. CONFIDENCE: Low"
    else:
        try:
            resp = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role":"system","content": TIM_PROMPT.format(region=region, refs=refs_md)},
                    {"role":"user","content": q}
                ],
                temperature=0.1,
                max_tokens=600
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print("Groq error:", e)
            answer = f"1. DIRECT ANSWER:\nError.\n\n2. EXPLANATION:\n{str(e)}\n\n3. REFERENCES:\n{refs_md}\n\n4. CONFIDENCE: Low"
    
    # Log query
    try:
        with open("/tmp/queries.log","a") as f:
            f.write(f"{datetime.utcnow().isoformat()},{region},{q}\n")
    except: pass
    
    return jsonify({"query": q, "region": region, "answer": answer, "sources": {"website": len(site_results), "pinecone": len(pc_results)}})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
