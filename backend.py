from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "PORT TEST OK - Ask TIM base works"}

@app.post("/ask")
def ask(q: dict):
    return {"answer": "test"}
