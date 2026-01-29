from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import ChatRequest, ChatResponse
from agent import run_agent

app = FastAPI(
    title="AI Agent API",
    version="1.0.0"
)

# Allow mobile apps / web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    reply, session_id = run_agent(
        request.message,
        request.session_id
    )
    return ChatResponse(
        reply=reply,
        session_id=session_id
    )

@app.get("/")
def health_check():
    return {"status": "ok"}
