"""
Ask TIM v5.3.2 - Matches your HTML test page
- Accepts {question, audience} from your frontend
- Consumer vs Professional modes
- Root / shows HTML, /health shows JSON
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
CORS(app) # Allows your HTML test page to POST

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
                print("Groq initialized")
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
                print(f"Pinecone initialized: {idx}")
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
            if ip and not ip.startswith('10.'):
                r = requests.get(f"http://ip-api.com/json/{ip}?fields=countryCode", timeout=1).json()
                country = r.get('countryCode', 'US')
        except: country = 'US'
    c = country.upper()
    if c in ['US','CA','MX','BR','AR','CL','CO']: return 'Americas'
    if c in ['GB','DE','FR','IT','ES','NL','SE','CH','IE']: return 'Europe'
    if c in ['IN','JP','CN','SG','AU','HK','KR','NZ']: return 'Asia-Pacific'
    return 'Americas'

def search_insuremaster(query):
    try:
        url = f"https://theinsuremaster.com/?s={requests.utils.quote(query)}"
        r = requests.get(url, timeout=5, headers={"User-Agent":"AskTIM/5.3"})
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

PROMPT_PRO = """You are Ask TIM for insurance professionals at theinsuremaster.com. USER REGION: {region}

Use technical insurance language. Cite policy forms, ISO endorsements, and {region} case law.

Format exactly:
1. DIRECT ANSWER:
[1-2 sentences]

2. EXPLANATION:
[3-4 sentences with technical detail]

3. REFERENCES:
{refs}

4. CONFIDENCE: High/Medium/Low
"""

PROMPT_CONSUMER = """You are Ask TIM for insurance consumers at theinsuremaster.com. USER REGION: {region}

Explain in plain English. No jargon, no citations. Tell them what it means and what to do.

Format exactly:
1. DIRECT ANSWER:
[simple]

2. EXPLANATION:
[plain language]

3. REFERENCES:
{refs}

4. CONFIDENCE: High/Medium/Low
"""

@app.route('/', methods=['GET','HEAD'])
def home():
    if request.method == 'HEAD':
        return '', 200
    q = request.args.get('q')
    if q:
        return ask()
    return jsonify({
        "service": "Ask TIM",
        "status": "live",
        "version": "5.3.2",
        "groq": bool(get_groq()),
        "pinecone": bool(get_pinecone())
    })

@app.route('/health')
def health():
    return home()

@app.route('/ask', methods=['GET','POST'])
def ask():
    data = request.json or {}
    # Accept your HTML names: question + audience
    q = request.args.get('q') or data.get('q') or data.get('question','')
    mode = request.args.get('mode') or data.get('mode') or data.get('audience','professional')

    if not q:
        return jsonify({"error":"No query", "answer":"Please provide a question"}), 400

    region = get_user_region()
    audience = 'consumer' if str(mode).lower() == 'consumer' else 'professional'

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
        answer = f"1. DIRECT ANSWER:\nGroq key missing.\n\n2. EXPLANATION:\nAdd GROQ_API_KEY in Render.\n\n3. REFERENCES:\n{refs_md}\n\n4. CONFIDENCE: Low"
    else:
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content": prompt.format(region=region, refs=refs_md)},
                    {"role":"user","content": q}
                ],
                temperature=0.3 if audience=='consumer' else 0.1,
                max_tokens=700
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"1. DIRECT ANSWER:\nError\n\n2. EXPLANATION:\n{str(e)}\n\n3. REFERENCES:\n{refs_md}\n\n4. CONFIDENCE: Low"

    return jsonify({
        "query": q,
        "mode": audience,
        "region": region,
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT",5000)))
