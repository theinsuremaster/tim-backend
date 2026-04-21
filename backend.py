"""
Ask TIM v5.3.1 - HTML test page + Consumer/Professional modes
- Root / returns HTML test page (fixes your test page)
- /health returns JSON for Render
- /ask supports ?mode=consumer or ?mode=professional
- Uses tim-knowledge Pinecone index
"""

import os
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import traceback

app = Flask(__name__)
CORS(app)

_groq_client = None
_pinecone_index = None
_embedder = None

def get_groq():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if key:
                _groq_client = Groq(api_key=key)
        except Exception as e:
            print("Groq error:", e)
    return _groq_client

def get_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        try:
            from pinecone import Pinecone
            key = os.getenv("PINECONE_API_KEY")
            if key:
                pc = Pinecone(api_key=key)
                idx = os.getenv("PINECONE_INDEX", "tim-knowledge")
                _pinecone_index = pc.Index(idx)
        except Exception as e:
            print("Pinecone error:", e)
    return _pinecone_index

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding('BAAI/bge-small-en-v1.5')
        except: pass
    return _embedder

def get_user_region():
    country = request.headers.get('CF-IPCountry', 'US')
    if not country or country == 'XX':
        try:
            ip = request.headers.get('X-Forwarded-For', '').split(',')[0]
            if ip:
                r = requests.get(f"http://ip-api.com/json/{ip}?fields=countryCode", timeout=1).json()
                country = r.get('countryCode', 'US')
        except: country = 'US'
    c = country.upper()
    if c in ['US','CA','MX','BR','AR']: return 'Americas'
    if c in ['GB','DE','FR','IT','ES','NL','SE']: return 'Europe'
    if c in ['IN','JP','CN','SG','AU']: return 'Asia-Pacific'
    return 'Americas'

def search_insuremaster(query):
    try:
        url = f"https://theinsuremaster.com/?s={requests.utils.quote(query)}"
        r = requests.get(url, timeout=5, headers={"User-Agent":"AskTIM"})
        soup = BeautifulSoup(r.text, 'lxml')
        out = []
        for a in soup.select('article h2 a, article h3 a')[:3]:
            title = a.get_text(strip=True)
            link = a['href']
            price = ''
            if '/product/' in link:
                try:
                    p = requests.get(link, timeout=3)
                    pr = BeautifulSoup(p.text,'lxml').select_one('.amount')
                    if pr: price = f" - {pr.get_text(strip=True)}"
                except: pass
            out.append({"title": title, "url": link, "price": price})
        return out
    except: return []

def search_pinecone(query, region):
    idx = get_pinecone()
    emb = get_embedder()
    if not idx or not emb: return []
    try:
        vec = list(emb.embed([query]))[0].tolist()
        res = idx.query(vector=vec, top_k=3, include_metadata=True,
                       filter={"region":{"$in":[region.lower(),"global"]}})
        return [{"title": m['metadata'].get('title',''), "url": m['metadata'].get('url','')} 
                for m in res.get('matches',[])]
    except: return []

# Prompts with audience differentiation
PROMPT_PRO = """You are Ask TIM for insurance professionals at theinsuremaster.com. USER REGION: {region}

Use technical insurance language. Cite policy forms, endorsements, case law.

Format:
1. DIRECT ANSWER:
[1-2 sentences, technical]

2. EXPLANATION:
[3-4 sentences with {region} statutes/cases]

3. REFERENCES:
{refs}

4. CONFIDENCE: High/Medium/Low
"""

PROMPT_CONSUMER = """You are Ask TIM for insurance consumers at theinsuremaster.com. USER REGION: {region}

Explain in plain English, no jargon. No case citations. Focus on what it means for their policy and what to do next.

Format:
1. DIRECT ANSWER:
[1-2 simple sentences]

2. EXPLANATION:
[3-4 sentences in plain language for {region}]

3. REFERENCES:
{refs}

4. CONFIDENCE: High/Medium/Low
"""

@app.route('/', methods=['GET','HEAD'])
def home():
    if request.method == 'HEAD':
        return '', 200
    
    q = request.args.get('q')
    mode = request.args.get('mode', 'professional')
    
    # If query present, process it
    if q:
        return ask()
    
    # HTML test page (this is what you had before)
    groq_ok = '✓' if get_groq() else '✗'
    pc_ok = '✓' if get_pinecone() else '✗'
    
    return f"""
    <!DOCTYPE html>
    <html><head><title>Ask TIM Test</title>
    <style>body{{font-family:system-ui;padding:40px;max-width:800px;margin:auto}}
    input,select{{padding:10px;font-size:16px}} button{{padding:10px 20px}}
    .status{{background:#f0f0f0;padding:15px;border-radius:8px;margin-bottom:20px}}
    </style></head><body>
    <h1>Ask TIM v5.3.1</h1>
    <div class="status">
      <b>Status:</b> Live<br>
      <b>Groq:</b> {groq_ok} &nbsp; <b>Pinecone (tim-knowledge):</b> {pc_ok}
    </div>
    
    <form action="/" method="get">
      <input type="text" name="q" placeholder="Ask about coverage..." style="width:400px" required>
      <select name="mode">
        <option value="professional" {"selected" if mode=="professional" else ""}>Professional</option>
        <option value="consumer" {"selected" if mode=="consumer" else ""}>Consumer</option>
      </select>
      <button type="submit">Ask TIM</button>
    </form>
    
    <p style="margin-top:30px;color:#666">API: <a href="/health">/health</a> | 
    Example: <a href="/?q=what is an additional insured&mode=consumer">Consumer test</a> | 
    <a href="/?q=additional insured exclusion&mode=professional">Pro test</a></p>
    </body></html>
    """

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "version": "5.3.1",
        "groq": bool(get_groq()),
        "pinecone_index": os.getenv("PINECONE_INDEX", "tim-knowledge"),
        "pinecone_connected": bool(get_pinecone())
    })

@app.route('/ask', methods=['GET','POST'])
def ask():
    data = request.json or {}
    q = request.args.get('q') or data.get('q','')
    mode = request.args.get('mode') or data.get('mode','professional')
    if not q:
        return jsonify({"error":"No query"}), 400
    
    region = get_user_region()
    audience = 'consumer' if mode.lower() == 'consumer' else 'professional'
    
    site = search_insuremaster(q)
    pc = search_pinecone(q, region)
    
    refs = []
    for r in site[:2]:
        refs.append(f"- [{r['title']}{r.get('price','')}]({r['url']})")
    for r in pc[:1]:
        if r.get('url'): refs.append(f"- [{r['title']}]({r['url']})")
    if not refs: refs.append("- [The InsureMaster](https://theinsuremaster.com)")
    refs_md = "\n".join(refs)
    
    prompt = PROMPT_CONSUMER if audience == 'consumer' else PROMPT_PRO
    client = get_groq()
    
    if not client:
        answer = f"1. DIRECT ANSWER:\nSetup needed.\n\n2. EXPLANATION:\nAdd GROQ_API_KEY.\n\n3. REFERENCES:\n{refs_md}\n\n4. CONFIDENCE: Low"
    else:
        try:
            resp = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role":"system","content": prompt.format(region=region, refs=refs_md)},
                    {"role":"user","content": f"[{audience.upper()}] {q}"}
                ],
                temperature=0.2 if audience=='consumer' else 0.1,
                max_tokens=700
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"
    
    # Return HTML if browser, JSON if API
    if 'text/html' in request.headers.get('Accept','') and not request.path.startswith('/ask'):
        return f"<pre style='font-family:system-ui;white-space:pre-wrap;padding:40px'>{answer}</pre><p><a href='/'>← Back</a></p>"
    
    return jsonify({
        "query": q,
        "mode": audience,
        "region": region,
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT",5000)))
