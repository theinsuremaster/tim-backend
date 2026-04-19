from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str

# Lazy globals
_groq = None
_pc_index = None
_embedder = None

def get_services():
    global _groq, _pc_index, _embedder
    if _groq is None:
        from groq import Groq
        from pinecone import Pinecone
        from fastembed import TextEmbedding

        groq_key = os.getenv("GROQ_API_KEY")
        pc_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX", "tim-knowledge")

        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY in Render Environment")
        if not pc_key:
            raise ValueError("Missing PINECONE_API_KEY in Render Environment")

        _groq = Groq(api_key=groq_key)
        pc = Pinecone(api_key=pc_key)
        _pc_index = pc.Index(index_name)
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _groq, _pc_index, _embedder

@app.get("/")
def root():
    return {"status": "Ask TIM is ready to be deployed"}

@app.post("/ask")
def ask(q: Query):
    try:
        groq_client, index, embedder = get_services()
        q_vec = list(embedder.embed(q.question))[0].tolist()
        results = index.query(vector=q_vec, top_k=3, include_metadata=True)
        context = "\n".join([m.metadata.get('text','') for m in results.matches if m.metadata])

        prompt = f"Use this context to answer:\n{context}\n\nQuestion: {q.question}"
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=512
        )
        return {"answer": resp.choices[0].message.content}
    except Exception as e:
        return {"error": str(e), "hint": "Check Render Environment variables and Pinecone index"}
