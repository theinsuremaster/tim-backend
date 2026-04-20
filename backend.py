import os, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

VERSION = "Ask TIM v4.3"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tim-knowledge")
embed = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"

COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")

TIERS = {
    "free": {"web_results": 10, "case_results": 3, "max_tokens": 1500},
    "pro": {"web_results": 25, "case_results": 5, "max_tokens": 2500},
    "enterprise": {"web_results": 25, "case_results": 8, "max_tokens": 3000}
}

def get_internal(question):
    try:
        vec = list(embed.embed(question))[0].tolist()
        results = index.query(vector=vec, top_k=8, include_metadata=True)
        return "\n\n".join([m.metadata.get('text','')[:800] for m in results.matches])
    except:
        return ""

def search_engines(question, limit):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    results = []
    seen = set()

    # 1. DuckDuckGo
    try:
        r = requests.get(f"https://html.duckduckgo.com/html/?q={question}", headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')
        for res in soup.select('.result')[:limit]:
            title = res.select_one('.result__title')
            snippet = res.select_one('.result__snippet')
            link = res.select_one('.result__url')
            if title and link:
                href = link.get('href','')
                if href and href not in seen and href.startswith('http'):
                    results.append({"title": title.get_text(strip=True), "snippet": snippet.get_text(strip=True)[:280] if snippet else "", "url": href})
                    seen.add(href)
    except: pass

    # 2. Brave
    if len(results) < limit:
        try:
            r = requests.get(f"https://search.brave.com/search?q={question}", headers=headers, timeout=8)
            soup = BeautifulSoup(r.text, 'html.parser')
            for item in soup.select('div.snippet')[:limit-len(results)]:
                a = item.find_previous('a')
                if a and a.get('href','').startswith('http'):
                    href = a['href']
                    if href not in seen:
                        results.append({"title": a.get_text(strip=True)[:80], "snippet": item.get_text(strip=True)[:280], "url": href})
                        seen.add(href)
        except: pass

    # 3. Bing - restored
    if len(results) < limit:
        try:
            r = requests.get(f"https://www.bing.com/search?q={question}", headers=headers, timeout=8)
            soup = BeautifulSoup(r.text, 'html.parser')
            for li in soup.select('li.b_algo')[:limit-len(results)]:
                a = li.select_one('h2 a')
                p = li.select_one('.b_caption p')
                if a and a.get('href'):
                    href = a['href']
                    if href not in seen and href.startswith('http'):
                        results.append({"title": a.get_text(strip=True), "snippet": p.get_text(strip=True)[:280] if p else "", "url": href})
                        seen.add(href)
        except: pass

    return results[:limit]

def get_web(question, tier, audience):
    return search_engines(question, TIERS[tier]["web_results"])

def get_caselaw(question, audience, tier):
    if audience!= "professional": return []
    if not any(k in question.lower() for k in ['coverage','policy','exclusion','claim','duty','bad faith','indemnity']): return []

    limit = TIERS[tier]["case_results"]
    cases = []
    is_uk = any(w in question.lower() for w in ['uk','england','scotland','wales'])

    try:
        if is_uk:
            r = requests.get(f"https://html.duckduckgo.com/html/?q={question} insurance site:bailii.org", headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
            soup = BeautifulSoup(r.text, 'html.parser')
            for res in soup.select('.result')[:limit]:
                link = res.select_one('.result__url'); title = res.select_one('.result__title')
                if link and 'bailii' in link.get('href',''):
                    cases.append({"title": title.get_text(strip=True), "url": link.get('href','')})
        else:
            if COURTLISTENER_TOKEN:
                headers = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}
                r = requests.get("https://www.courtlistener.com/api/rest/v4/search/", headers=headers, params={"q": f"{question} insurance", "type": "o", "order_by": "score desc"}, timeout=10)
                if r.status_code == 200:
                    for item in r.json().get('results', [])[:limit]:
                        name = item.get('caseName',''); url = f"https://www.courtlistener.com{item.get('absolute_url','')}"
                        cases.append({"title": f"{name} {item.get('citation','')}".strip(), "url": url})
    except: pass
    return cases

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question","")
    audience = data.get("audience","consumer")
    tier = data.get("tier","free")

    if not question: return jsonify({"error":"No question"}), 400

    blocked = ['diagnose','prescribe','medical advice','hipaa','attorney client']
    if any(b in question.lower() for b in blocked) and 'insurance' not in question.lower():
        return jsonify({"answer": "Ask TIM specializes in insurance only. See https://theinsuremaster.com/disclaimer"})

    web_results = get_web(question, tier, audience)
    internal = get_internal(question) if audience == "professional" else ""
    cases = get_caselaw(question, audience, tier) if audience == "professional" else []

    references = []
    for w in web_results:
        if w['url']: references.append(f"[{w['title']}]({w['url']})")
    for c in cases: references.append(f"[{c['title']}]({c['url']})")

    confidence = "High"
    if audience == "professional" and len(cases) == 0 and 'coverage' in question.lower():
        confidence = "Medium - limited case law found"

    if audience == "consumer":
        voice = "You are Ask TIM for consumers. Answer in plain English, paraphrased. Example: 'Are there any benefits of bundling home and auto insurance with one insurer, or can I insure them separately' — explain pros/cons. Use ## headings, bullets. No links in body."
    else:
        voice = "You are Ask TIM for professionals. Paraphrase internal knowledge. Do not cite Pinecone publications. Cite case law properly by legal name only."

    system = f"{voice}\nNever copy verbatim. Only cite case law."

    web_text = "\n".join([f"{w['title']}: {w['snippet']}" for w in web_results[:10]])
    user_prompt = f"QUESTION: {question}\n\nINTERNAL:\n{internal[:2000]}\n\nWEB:\n{web_text}\n\nCASES:\n{chr(10).join([c['title'] for c in cases])}"

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":system},{"role":"user","content":user_prompt}],
        temperature=0.3,
        max_tokens=TIERS[tier]["max_tokens"]
    )
    answer = completion.choices[0].message.content

    if references:
        answer += "\n\n### References\n" + "\n".join([f"- {ref}" for ref in references[:15]])
    if audience == "professional":
        answer += f"\n\n*Confidence: {confidence}*"
    answer += "\n\n---\n[Disclaimer](https://theinsuremaster.com/disclaimer)"

    return jsonify({"answer": answer, "audience": audience})

@app.route("/")
def home():
    return jsonify({"status": VERSION})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)))
