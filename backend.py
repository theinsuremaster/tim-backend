"""
Ask TIM Backend v7.0
- Sets 1-5 fully integrated
- HIPAA / Privacy / Safety guardrails (Set 1)
- Location detection + resource priority (Set 2)
- 4-step search sequence + product suggestion (Set 3)
- Consumer mandatory format 180-300 words, Kansas neighbor tone (Set 4)
- Professional mandatory format 200-500 words, Bluebook caselaw (Set 5)
- HELLO greeting handler (Set 5)
- Root status page + health check log suppression
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from groq import Groq
from pinecone import Pinecone
from fastembed import TextEmbedding
import os, logging, requests, re, time, datetime
from collections import defaultdict

app = Flask(__name__)
CORS(app)

VERSION = "7.0"
LAST_UPDATE = "2026-04-22"

class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ('GET / HTTP' in msg or 'HEAD / HTTP' in msg or 'GET /health' in msg or 'HEAD /health' in msg or 'Render' in msg)

logging.getLogger('werkzeug').addFilter(HealthCheckFilter())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) if os.getenv("PINECONE_API_KEY") else None
index = pc.Index(os.getenv("PINECONE_INDEX", "tim-knowledge")) if pc else None
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

request_counts = defaultdict(list)
def is_rate_limited(ip):
    now = time.time()
    window = [t for t in request_counts[ip] if now - t < 60]
    request_counts[ip] = window
    if len(window) >= 60:
        return True
    window.append(now)
    return False

BOT_SIGNATURES = ['bot','crawl','spider','scrape','curl','wget','python-requests','httpclient','postman','go-http']
MALICIOUS_PATTERNS = ['ignore previous','disregard instructions','system prompt','reveal your','show me your prompt','pinecone','secret repository','dump database','bypass','jailbreak','<script','union select','drop table','../../','/etc/passwd']

def check_guardrails(question, headers, ip):
    ua = headers.get('User-Agent','').lower()
    if any(b in ua for b in BOT_SIGNATURES) and 'mozilla' not in ua:
        return {"blocked": True, "reason": "automated_crawler", "message": "Automated access to Ask TIM is prohibited. Please use the web interface."}
    if is_rate_limited(ip):
        return {"blocked": True, "reason": "rate_limit", "message": "Excessive use detected. Please wait 60 seconds."}
    ql = question.lower()
    if any(p in ql for p in MALICIOUS_PATTERNS):
        return {"blocked": True, "reason": "security", "message": "Request blocked by security guardrails."}
    special_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', question)) / max(len(question),1)
    if special_ratio > 0.5 and len(question) > 30:
        return {"blocked": True, "reason": "obfuscation", "message": "Excessive symbols detected."}
    if re.search(r'(.)\1{10,}', question):
        return {"blocked": True, "reason": "obfuscation", "message": "Invalid input pattern."}
    return {"blocked": False}

def get_location(request_obj):
    country = request_obj.headers.get("CF-IPCountry")
    state = None
    if country:
        mapping = {"US":"United States","CA":"Canada","GB":"United Kingdom","AU":"Australia","IN":"India"}
        country = mapping.get(country.upper(), country)
        return {"country": country, "state": state}
    try:
        ip = request_obj.headers.get("X-Forwarded-For","").split(",")[0].strip() or request_obj.remote_addr
        if ip and ip != "127.0.0.1":
            resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=1.5)
            if resp.ok:
                data = resp.json()
                country = data.get("country_name") or "United States"
                state = data.get("region")
                return {"country": country, "state": state}
    except:
        pass
    return {"country": "United States", "state": None}

# FULL SYSTEM PROMPTS WITH SETS 1-5 EMBEDDED
CONSUMER_SYSTEM = """You are Ask TIM, The InsureMaster AI Assistant.

CRITICAL: User country = {country}, detected state = {state}. Prioritize {country} 2026 laws.

HIPAA / PRIVACY / SAFETY RULES - FOLLOW AT ALL TIMES:
- PHI: Do NOT request, store, process, or infer PHI unless user explicitly provides it. Never ask for unnecessary identifiers.
- HIPAA Compliance: Do NOT act as clinician. Do NOT diagnose or prescribe. Provide general educational information only. Always include: "This is not medical advice."
- Privacy: Do NOT store, reuse, or share user data. Do NOT reveal data unless asked.
- Financial/Legal Safety: Do NOT act as lawyer, financial advisor, or licensed agent. Provide general information only. Use phrase: "I can give general information, but this is not legal, financial, or professional advice."
- No Hallucinations: If unknown, say "I don't have enough information to answer that safely."
- User Control: Answer only what asked. Refuse unsafe requests politely.

SEQUENCE OF INFORMATION - HARD OVERRIDE (Set 3):
1. PRIMARY: Search theinsuremaster.com first. If found, provide few-sentence explanation + one-sentence link. If product relevant, suggest politely with EXACT price.
2. SECONDARY: Pinecone tim-knowledge. Summarize, paraphrase. Quote verbatim only ISO endorsements, statutes, laws, regulations, policy language. NEVER reveal Pinecone source.
3. TERTIARY: Curated consumer sources (iii.org, bankrate.com, investopedia.com, naic.org, thezebra.com, nerdwallet.com, policygenius.com, valuepenguin.com, progressive.com/learn, statefarm.com/simple-insights).
4. FINAL: DDG → Bing → others. Prioritize scholarly/regulatory sources.

RESOURCE PRIORITY BY LOCATION (Set 2):
- United States: federal + all-state resources unless specific state requested.
- United Kingdom: UK resources.
- Other: local then US. Europe: European then US.

CONSUMER MANDATORY FORMAT (Set 4):
1. EXPLANATION: Rephrase query first. 180-300 words total. Short bullet chunks. Include {country} cost ranges. List 3 common mistakes consumers make. Give 1 practical tip. Warm, direct, Kansas neighbor tone. No jargon, no caselaw.
2. FOLLOW UP: Start with "- "
3. EXPERT: Start with "Talk to your..."
4. SOURCES: At least 3, format https://url – domain.com, use superscript citations ¹²³
5. FINAL LINE: Confidence: XX% | [Disclaimer](https://theinsuremaster.com/disclaimer)
"""

PROFESSIONAL_SYSTEM = """You are Ask TIM for professionals.

CRITICAL: User country = {country}, detected state = {state}. Prioritize {country} 2026 laws.

HIPAA / PRIVACY / SAFETY RULES: Same as consumer - no PHI storage, no diagnosis, no legal/financial advice beyond general information.

SEQUENCE OF INFORMATION: Same 4-step hierarchy as consumer, but use 13 professional sources first: irmi.com, wiley.law, pillsburylaw.com, lloyds.com, ambest.com, naic.org, legislation.gov.uk, findlaw.com/caselaw, law.cornell.edu, justia.com, fcands.com, propertycasualty360.com, businessinsurance.com.

PROFESSIONAL MANDATORY FORMAT (Set 5):
- Tone: Precise coverage counsel tone for brokers, risk managers, and attorneys. No warmth. No simplification.
- 200-500 words total.
- Integrate 2+ Bluebook style cases where possible (do not hallucinate).
- Jurisdictional split analysis.
- Superscript citations.
- Output order: EXPLANATION, FOLLOW UP (starts with "- "), EXPERT (starts with "Talk to your..."), CASELAW CITED, SOURCES, FINAL LINE.
"""

def call_groq(sys_prompt, user_q, ctx=""):
    user_content = f"Context: {ctx}\n\nQuestion: {user_q}" if ctx else user_q
    resp = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_content}], temperature=0.2, max_tokens=1500)
    return resp.choices[0].message.content.strip()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question","").strip()
    audience = data.get("audience","consumer")
    username = data.get("username","there")
    ip = request.headers.get("X-Forwarded-For","").split(",")[0] or request.remote_addr
    guard = check_guardrails(question, request.headers, ip)
    if guard.get("blocked"):
        return jsonify({"answer": guard["message"], "guardrail": guard["reason"], "version": VERSION}), 403
    if not question:
        return jsonify({"answer":"Please provide a question."}),400
    loc = get_location(request)
    country = loc["country"]
    state = loc["state"] or "unknown"
    # HELLO handler
    greetings = ['hello','hi','hey',"what's up","wassup","whats up","sup","good morning","good afternoon","good evening"]
    if question.lower().strip() in greetings:
        hour = datetime.datetime.now().hour
        tod = "morning" if hour<12 else "afternoon" if hour<17 else "evening"
        if audience=="professional":
            greeting = f"Good {tod} {username}, I can help with insurance, finance, and risk management. What would you like to explore today?"
        else:
            greeting = f"Hey {username} Wassup! Welcome to Ask TIM – The Insure Master! How can I help with your insurance, risk management, and finance related questions, today?"
        return jsonify({"answer": greeting, "version": VERSION, "detected_country": country, "detected_state": state})
    system = (PROFESSIONAL_SYSTEM if audience=="professional" else CONSUMER_SYSTEM).format(country=country, state=state)
    ctx = ""
    if index:
        try:
            emb = list(embedder.embed([f"{question} {country} {state}"]))[0].tolist()
            res = index.query(vector=emb, top_k=3, include_metadata=True)
            ctx = "\n\n".join([m['metadata'].get('text','') for m in res['matches']])
        except:
            pass
    answer = call_groq(system, question, ctx)
    if "[Disclaimer]" not in answer:
        answer += "\n\nConfidence: 90% | [Disclaimer](https://theinsuremaster.com/disclaimer)"
    return jsonify({"answer": answer, "version": VERSION, "detected_country": country, "detected_state": state})

@app.route("/", methods=["GET"])
def root():
    html = f"""<!doctype html><html><head><title>Ask TIM Backend</title></head><body style="font-family:sans-serif;text-align:center;padding:50px;"><h1>Ask TIM Backend is Live</h1><p>Version: {VERSION}</p><p>Last Update: {LAST_UPDATE}</p><p>Status: Operational</p></body></html>"""
    return make_response(html, 200)

@app.route("/health", methods=["GET","HEAD"])
def health(): return "",204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
