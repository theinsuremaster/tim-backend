from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq
from pinecone import Pinecone

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "tim-knowledge"))

MODEL = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def root():
    return {"status": "Ask TIM backend running"}

@app.post("/ask")
def ask(q: Query):
    # 1. Retrieve from Pinecone (simple vector search - using query as text for demo)
    # In production you'd embed the question first
    try:
        results = index.query(
            vector=[0.0]*1024,  # placeholder - replace with real embedding
            top_k=q.top_k,
            include_metadata=True
        )
        context = "\n\n".join([m.metadata.get("text","") for m in results.matches[:3]])
    except Exception as e:
        context = "No knowledge base yet."

    prompt = f"""You are Ask TIM, a helpful expert assistant. Use the context below to answer.

Context:
{context}

Question: {q.question}

Answer concisely and helpfully:"""

    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    
    return {"answer": resp.choices[0].message.content, "context_used": bool(context)}
