# tim_bot_v5_7_11.py
# The Insurance Master — Professional Mode
# Pinecone + Groq + 18-source verification

import os, requests, re
from datetime import datetime
from groq import Groq
from pinecone import Pinecone

# --- CONFIG ---
GROQ_MODEL = "llama3-70b-8192"
PINECONE_INDEX = "theinsuremaster"
VERSION = "5.7.11"

TIER_1_PINE = True # hidden vector search

SOURCES = {
    "tier1_web": ["theinsuremaster.com"],
    "tier2_insurance": [
        "iii.org",
        "naic.org", "content.naic.org",
        "irmi.com",
        "legalclarity.org", # NEW
        "ambest.com",
        "fcands.com",
        "law.cornell.edu",
        "iso.com",
        "aaisdirect.com"
    ],
    "tier2_finance": [ # NEW
        "investopedia.com",
        "finance.yahoo.com",
        "google.com/finance",
        "barrons.com",
        "bloomberg.com"
    ],
    "tier3_trade": [
        "propertycasualty360.com",
        "insurancejournal.com"
    ],
    "caselaw": [
        "courtlistener.com",
        "law.justia.com",
        "courts.ca.gov",
        "nycourts.gov",
        "txcourts.gov",
        "floridasupremecourt.org",
        "illinoiscourts.gov"
    ]
}

# --- CLIENTS ---
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

# --- CORE FUNCTIONS ---
def pinecone_search(query, top_k=5):
    """Tier 1 hidden - returns chunks, not shown in Sources"""
    emb = pc.inference.embed(model="multilingual-e5-large", inputs=[query])[0].values
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [m.metadata['text'] for m in res.matches if m.score > 0.78]

def web_search(query, site):
    """Simple SerpAPI wrapper - returns first 200 OK result"""
    # placeholder - replace with your search implementation
    url = f"https://serpapi.com/search?q=site:{site}+{query}"
    #... implement...
    return None

def verify_200(url):
    try:
        r = requests.head(url, timeout=3, allow_redirects=True)
        return r.status_code == 200
    except:
        return False

def find_caselaw(concept):
    cases = []
    for site in SOURCES['caselaw']:
        # search logic
        # verify citation pattern \d+ [A-Z]\.\w+ \d+
        pass
    # return list of dicts: {"cite": "Morgan Stanley Group Inc. v. New England...", "verified": True}
    return cases[:3]

def format_professional(question, pine_context, web_sources, cases):
    prompt = f"""
    You are TIM v{VERSION}. Answer professionally in 5 paragraphs.
    Use this internal context (from Pinecone, do not cite): {pine_context}
    Web sources: {web_sources}
    Caselaw: {cases}

    Rules:
    - Start with plain meaning, then extrinsic evidence, then contra proferentem
    - Cite cases with <sup>n</sup>
    - Do not hallucinate cases
    """
    resp = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

def build_sources_block(sources):
    block = "**Sources & What Was Searched**\n"
    for i, s in enumerate(sources, 1):
        if verify_200(s['url']):
            block += f'{i}. **Searched:** "{s["query"]}" on {s["site"]}\n'
            block += f' **Found:** [{s["title"]}]({s["url"]})\n'
            block += f' **Scraped:** "{s["snippet"][:200]}..."\n\n'
    return block

def build_caselaw_block(cases):
    block = "**Caselaw Cited**\n"
    for i, c in enumerate(cases, 1):
        # NO LINKS per your rule
        block += f'{i}. *{c["cite"]}*\n'
    return block

# --- MAIN ---
def answer(question):
    # 1. Pinecone (hidden)
    pine_ctx = pinecone_search(question)

    # 2. Tiered web search
    web_sources = []
    for site in SOURCES['tier1_web'] + SOURCES['tier2_insurance']:
        result = web_search(question, site)
        if result and verify_200(result['url']):
            web_sources.append(result)
        if len(web_sources) >= 3: break

    # 3. Finance sites only if needed
    if any(k in question.lower() for k in ["rate", "dividend", "stock", "market", "interest"]):
        for site in SOURCES['tier2_finance']:
            result = web_search(question, site)
            if result: web_sources.append(result)

    # 4. Caselaw
    cases = find_caselaw(question)

    # 5. Generate
    body = format_professional(question, pine_ctx, web_sources, cases)

    # 6. Assemble
    output = f"**PROFESSIONAL ANSWER**\n\n**Question:** {question}\n\n**Answer:**\n{body}\n\n"
    output += build_sources_block(web_sources)
    output += "\n" + build_caselaw_block(cases)
    output += f"\n---\n*TIM v{VERSION}*"

    return output

# --- EXAMPLE ---
if __name__ == "__main__":
    q = "How do courts determine ambiguity in insurance clauses?"
    print(answer(q))
