from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "tim-knowledge"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedder

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "Ask TIM with Pinecone"}

@app.post("/ask")
def ask(q: Query):
    embedder = get_embedder()
    q_vec = list(embedder.embed(q.question))[0].tolist()
    results = index.query(vector=q_vec, top_k=3, include_metadata=True)
    context = "\n".join([m.metadata.get('text','') for m in results.matches])
    prompt = f"Context:\n{context}\n\nQuestion: {q.question}"
    resp = groq_client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=512)
    return {"answer": resp.choices[0].message.content}
