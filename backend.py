"""
Ask TIM Backend v6.0.14.5
- Model: llama-3.3-70b-versatile
- Auto-detect country
- Guardrails: topic scope + anti-bot/attack protection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding
import os, logging, requests, re, time
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# === LOG FILTER ===
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ('GET / HTTP' in msg or 'GET /health' in msg or 'Render/1.0' in msg)
logging.getLogger('werkzeug').addFilter(HealthCheckFilter())

# === CONFIG ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) if os.getenv("PINECONE_API_KEY") else None
index = pc.Index(os.getenv("PINECONE_INDEX", "tim-knowledge")) if pc else None
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# === RATE LIMIT ===
request_counts = defaultdict(list)
def is_rate_limited(ip):
    now = time.time()
    window = [t for t in request_counts[ip] if now - t < 60]
    request_counts[ip] = window
    if len(window) >= 60:  # 60 requests/minute
        return True
    window.append(now)
    return False

# === GUARDRAILS ===
ALLOWED_TOPICS = ['insurance','insure','policy','premium','claim','coverage','deductible','liability','underwriter','risk','risk management','finance','financial','invest','retirement','annuity','actuarial','reinsurance','broker','agent']

BOT_SIGNATURES = ['bot','crawl','spider','scrape','curl','wget','python-requests','httpclient','postman','go-http']

MALICIOUS_PATTERNS = [
    'ignore previous','disregard instructions','system prompt','reveal your','show me your prompt',
    'pinecone','secret repository','dump database','bypass','jailbreak','<script','union select',
    'drop table','../../','/etc/passwd'
]

def check_guardrails(question, headers, ip):
    # 1. Bot detection
    ua = headers.get('User-Agent','').lower()
    if any(b in ua for b in BOT_SIGNATURES) and 'mozilla' not in ua:
        return {"blocked": True, "reason": "automated_crawler", "message": "Automated access to Ask TIM is prohibited. Please use the web interface."}
    
    # 2. Rate limit
    if is_rate_limited(ip):
        return {"blocked": True, "reason": "rate_limit", "message": "Excessive use detected. Please wait 60 seconds."}
    
    # 3. Attack patterns
    ql = question.lower()
    if any(p in ql for p in MALICIOUS_PATTERNS):
        return {"blocked": True, "reason": "security", "message": "Request blocked by security guardrails."}
    
    # 4. Typographical hack detection
    special_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', question)) / max(len(question),1)
    if special_ratio > 0.5 and len(question) > 30:
        return {"blocked": True, "reason": "obfuscation", "message": "Excessive symbols detected."}
    if re.search(r'(.){10,}', question):  # repeated chars
        return {"blocked": True, "reason": "obfuscation", "message": "Invalid input pattern."}
    
    # 5. Topic scope
    is_on_topic = any(t in ql for t in ALLOWED_TOPICS)
    persist = 'persist' in ql or 'answer anyway' in ql or 'continue' in ql
    
    if not is_on_topic and not persist:
        return {"blocked": False, "warning": True, "reason": "off_topic"}
    
    return {"blocked": False}

def get_country_auto(request_obj):
    country = request_obj.headers.get("CF-IPCountry")
    if country:
        mapping = {"US":"United States","CA":"Canada","GB":"United Kingdom","AU":"Australia","IN":"India"}
        return mapping.get(country.upper(), country)
    try:
        ip = request_obj.headers.get("X-Forwarded-For","").split(",")[0].strip() or request_obj.remote_addr
        if ip and ip != "127.0.0.1":
            resp = requests.get(f"https://ipapi.co/{ip}/country_name/", timeout=1.2)
            if resp.ok: return resp.text.strip()
    except: pass
    return "United States"

CONSUMER_SYSTEM = """You are Ask TIM, The InsureMaster AI Assistant for insurance, risk management, and finance ONLY.

CRITICAL: User country = {country}. Prioritize {country} 2026 laws.

FORMAT: [same as v6.0.14.4]
1. EXPLANATION (+25%): chunks, bullets, {country} cost, 3 mistakes, 1 tip
2. FOLLOW-UP: "- " research only
3. EXPERT: "Talk to your..."
4. SOURCES: 3 as https://url – domain.com
5. FINAL: Confidence: XX% | [Disclaimer](https://theinsuremaster.com/disclaimer)
"""

PROFESSIONAL_SYSTEM = """You are Ask TIM for insurance/risk/finance professionals ONLY. Country={country}."""

def call_groq(sys, q, ctx=""):
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":sys},{"role":"user","content":f"Context:{ctx}

{q}" if ctx else q}],
        temperature=0.2, max_tokens=1500
    )
    return resp.choices[0].message.content.strip()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question","").strip()
    audience = data.get("audience","consumer")
    ip = request.headers.get("X-Forwarded-For","").split(",")[0] or request.remote_addr
    
    # Guardrails
    guard = check_guardrails(question, request.headers, ip)
    if guard.get("blocked"):
        return jsonify({"answer": guard["message"], "guardrail": guard["reason"], "version":"6.0.14.5"}), 403
    
    if guard.get("warning"):
        return jsonify({
            "answer": "⚠️ Ask TIM is designed exclusively for insurance, risk management, and finance questions for consumers and professionals.

Your question appears outside this scope. Please rephrase to relate to insurance, risk, or financial planning.

If you want me to answer anyway, reply with the same question including the word 'persist'.",
            "guardrail": "off_topic_warning",
            "version": "6.0.14.5"
        })
    
    if not question:
        return jsonify({"answer":"Please provide a question."}),400
    
    country = get_country_auto(request)
    system = (PROFESSIONAL_SYSTEM if audience=="professional" else CONSUMER_SYSTEM).format(country=country)
    
    # RAG
    ctx = ""
    if index:
        try:
            emb = list(embedder.embed([f"{question} {country}"]))[0].tolist()
            res = index.query(vector=emb, top_k=3, include_metadata=True)
            ctx = "\n\n".join([m['metadata'].get('text','') for m in res['matches']])
        except: pass
    
    answer = call_groq(system, question, ctx)
    if "[Disclaimer]" not in answer:
        answer += "\n\nConfidence: 90% | [Disclaimer](https://theinsuremaster.com/disclaimer)"
    
    return jsonify({"answer": answer, "version":"6.0.14.5", "detected_country": country})

@app.route("/", methods=["GET","HEAD"]): 
    def root(): return "",204
@app.route("/health", methods=["GET","HEAD"]):
    def health(): return "",204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
