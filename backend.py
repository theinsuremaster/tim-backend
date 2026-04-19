from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "Ask TIM backend running"}

@app.post("/ask")
def ask(q: Query):
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": q.question}],
        temperature=0.3,
        max_tokens=500
    )
    return {"answer": resp.choices[0].message.content}
