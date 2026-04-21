# backend.py - v5.7.4b - Dynamic pricing per item
import os, requests, random, re
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tim-knowledge")
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")

pinecone_index = None
if PINECONE_API_KEY:
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX)
    except: pass

PRIORITY_SOURCES = ["naic.org","rims.org","theinstitutes.org","ambest.com","insurancejournal.com","propertycasualty360.com","businessinsurance.com","riskandinsurance.com","iii.org","irmi.com"]
USER_AGENTS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"]

@app.route("/health")
def health():
    return jsonify({"status":"ok","version":"5.7.4b"})

def detect_tim_content(url, title, snippet):
    """Auto-detect type, extract price if mentioned"""
    url_lower = url.lower()
    title_lower = title.lower()

    # Detect type
    if "/checklist/" in url_lower or "checklist" in title_lower:
        content_type = "checklist"
    elif "/course/" in url_lower or "/training/" in url_lower or "course" in title_lower:
        content_type = "course"
    elif "/premium/" in url_lower or "/pro/" in url_lower:
        content_type = "premium article"
    else:
        content_type = "article"

    # Try to extract price from snippet or title
    price = ""
    price_match = re.search(r'\$(\d+)', snippet + " " + title)
    if price_match:
        price = f"${price_match.group(1)}"
    elif "free" in snippet.lower() or content_type == "article":
        price = "Free"
    elif content_type == "course":
        price = "TIM Pro" # default, can be overridden by page scrape
    # else leave blank - you'll set in CMS

    return content_type, price

def search_theinsuremaster(question):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        url = f"https://www.bing.com/search?q={question}+site:theinsuremaster.com"
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, 'lxml')
        results = []
        for item in soup.select('li.b_algo')[:5]:
            title = item.select_one('h2'); snippet = item.select_one('.b_caption p') or item.select_one('p'); link = item.select_one('a')
            if snippet and link:
                href = link['href']
                title_text = title.get_text(strip=True) if title else ""
                snippet_text = snippet.get_text(strip=True)
                content_type, price = detect_tim_content(href, title_text, snippet_text)
                results.append({
                    "engine":"theinsuremaster.com",
                    "title":title_text,
                    "snippet":snippet_text[:400],
                    "url":href,
                    "source":"theinsuremaster.com",
                    "content_type":content_type,
                    "price":price
                })
        return results
    except: return []

def search_pinecone(question, top_k=3):
    if not pinecone_index: return []
    try:
        import numpy as np
        dummy_vector = list(np.random.rand(1536))
        response = pinecone_index.query(vector=dummy_vector, top_k=top_k, include_metadata=True)
        results = []
        for m in response.get("matches", []):
            meta = m.get("metadata", {})
            # Pull price from Pinecone metadata if you store it
            results.append({
                "engine":"Pinecone",
                "title":meta.get("title","Internal"),
                "snippet":meta.get("text","")[:400],
                "url":meta.get("url",""),
                "source":"tim-knowledge",
                "content_type":meta.get("type","internal"),
                "price":meta.get("price","") # store individual prices in Pinecone
            })
        return results
    except: return []

def search_bing_html(query, site=None):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        q = f"{query}+site:{site}" if site else query
        r = requests.get(f"https://www.bing.com/search?q={q}&count=5", headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, 'lxml')
        results = []
        for item in soup.select('li.b_algo')[:3]:
            title = item.select_one('h2'); snippet = item.select_one('.b_caption p') or item.select_one('p'); link = item.select_one('a')
            if snippet:
                results.append({"engine":"Bing","title":title.get_text(strip=True) if title else "","snippet":snippet.get_text(strip=True)[:400],"url":link['href'] if link else "","source":site or "web","content_type":"web article","price":""})
        return results
    except: return []

def search_caselaw(question, max_cases=5):
    if not COURTLISTENER_TOKEN: return []
    try:
        headers = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}
        r = requests.get("https://www.courtlistener.com/api/rest/v3/search/", headers=headers, params={"q":question,"type":"o","order_by":"score desc","page_size":max_cases}, timeout=8)
        cases = []
        if r.status_code == 200:
            for item in r.json().get("results", [])[:max_cases]:
                if item.get("caseName"):
                    cases.append({"case":item["caseName"],"citation":f"{item['caseName']}, {item.get('dateFiled','')[:10]}","court":item.get("court",""),"snippet":item.get("snippet","")[:300],"url":f"https://www.courtlistener.com{item.get('absolute_url','')}","date":item.get("dateFiled","")[:10]})
        return cases
    except: return []

def calculate_confidence(sources, has_tim, has_pinecone, has_caselaw, mode):
    score = 50
    if has_tim: score += 20
    if has_pinecone: score += 15
    if has_caselaw: score += 15
    score += min(len(sources)*3,15)
    if any('2024' in s.get('snippet','') or '2025' in s.get('snippet','') for s in sources): score+=5
    if mode=="professional" and has_caselaw: score+=10
    score = min(score,100)
    label = "Very High" if score>=85 else "High" if score>=70 else "Medium" if score>=55 else "Low"
    return {"score":score,"label":label}

def rewrite_with_groq(question, sources, cases, mode):
    if not sources:
        return "I don't want to give you an inaccurate answer on this one. If you can share a bit more detail, I can point you in the right direction or explain the options clearly."
    if not GROQ_API_KEY: return sources[0]['snippet']

    context = "\n\n".join([f"{s['source']}: {s['title']}\n{s['snippet']}" for s in sources[:4]])
    caselaw_context = ""
    if cases and mode=="professional":
        caselaw_context = "\n\nRelevant Caselaw:\n" + "\n".join([f"- {c['case']} ({c['date']}): {c['snippet'][:150]}" for c in cases[:3]])

    tim_mentions = ""
    tim_items = [s for s in sources if 'theinsuremaster' in s['source']]
    if tim_items:
        tim_mentions = "\n\nAvailable TIM resources:\n" + "\n".join([f"- {s['title']} ({s['content_type']}{', '+s['price'] if s['price'] else ''})" for s in tim_items[:3]])

    style_guide = """You are TIM, an experienced insurance producer. Conversational, direct, plain language. Adapt structure to question. Mention available TIM articles, checklists, or courses naturally when relevant."""

    prompt = f"{style_guide}\n\nQuestion: {question}\nMode: {mode}\n\nResearch:\n{context}{caselaw_context}{tim_mentions}\n\nWrite in natural TIM voice."

    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":prompt}],"temperature":0.7,"max_tokens":750}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=15)
        if r.status_code==200: return r.json()['choices'][0]['message']['content'].strip()
    except: pass
    return sources[0]['snippet']

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question","").strip()
    mode = data.get("mode","consumer")
    if not question: return jsonify({"error":"No question"}),400

    all_sources = []
    tim_results = search_theinsuremaster(question)
    all_sources.extend(tim_results)

    pinecone_results = search_pinecone(question)
    all_sources.extend(pinecone_results)

    if len(all_sources) < 2:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(search_bing_html, question, site) for site in PRIORITY_SOURCES[:5]]
            for f in futures:
                try: all_sources.extend(f.result(timeout=8))
                except: pass
        if len(all_sources) < 2:
            all_sources.extend(search_bing_html(question, None))

    cases = search_caselaw(question, max_cases=5) if mode=="professional" else []

    natural_answer = rewrite_with_groq(question, all_sources, cases, mode)
    confidence = calculate_confidence(all_sources, len(tim_results)>0, len(pinecone_results)>0, len(cases)>0, mode)

    if len(all_sources) == 0 or confidence['score'] < 40:
        return jsonify({
            "answer": natural_answer + "\n\n*Click here to view our Disclaimer.*",
            "sources": [], "caselaw": [], "confidence_index": confidence,
            "needs_clarification": True, "version": "5.7.4b"
        })

    # Format sources - price shown only if found
    source_lines = []
    for s in all_sources[:5]:
        if 'theinsuremaster' in s['source']:
            price_part = f", {s['price']}" if s['price'] else ""
            type_part = f" ({s['content_type']}{price_part})"
            line = f"{s['title']} - {s['source']} [{s['url']}]{type_part}"
        else:
            line = f"{s['title'][:70]} - {s['source']}"
        source_lines.append(f"• {line}")

    if cases and mode=="professional":
        source_lines.append("\n**Caselaw:**")
        for c in cases: source_lines.append(f"• {c['case']} ({c['date']}) - {c['url']}")

    full_answer = f"{natural_answer}\n\n---\n**Sources:**\n" + "\n".join(source_lines)
    full_answer += f"\n\n**Confidence Index:** {confidence['score']}/100 ({confidence['label']})"

    return jsonify({
        "answer": full_answer + "\n\n*Click here to view our Disclaimer.*",
        "sources": [s['source'] for s in all_sources],
        "caselaw": cases,
        "confidence_index": confidence,
        "tim_content_found": len(tim_results)>0,
        "version": "5.7.4b"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)))
