from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY or not GROQ_API_KEY:
    raise RuntimeError("❌ Missing API keys! Create a .env file with GEMINI_API_KEY and GROQ_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
groq_client  = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="MarketMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    mode: str = "chat"

class LeadRequest(BaseModel):
    company: str
    industry: str
    interactions: int
    budget_signal: str

SYSTEM_PROMPTS = {
    "chat": """You are MarketMind, an expert Generative AI assistant for sales and marketing teams.
You help with campaigns, lead insights, sales pitches, and market analysis.
Be concise, actionable, and data-driven. Use emojis sparingly for readability.""",

    "campaign": """You are MarketMind's Campaign Generator.
Generate a complete marketing campaign including:
- A catchy campaign name & tagline
- Email subject line and short body
- One LinkedIn post
- One Twitter/X post
- Expected KPIs (reach, CTR estimate)
Be specific, creative, and conversion-focused.""",

    "pitch": """You are MarketMind's Sales Pitch Writer.
Create a personalized, compelling sales pitch tailored to the audience described.
Include: hook, pain point, solution, proof/ROI, and a clear call to action.
Keep it under 200 words. Professional but human.""",

    "leads": """You are MarketMind's Lead Intelligence Engine.
Analyze the lead details provided and return:
- A lead score out of 100
- Priority level (HOT / WARM / COLD)
- 3 specific recommended next actions
- Estimated deal close probability %
Be analytical and direct.""",
}

def ask_gemini(prompt: str, system: str) -> str:
    full = f"{system}\n\nUser: {prompt}"
    response = gemini_model.generate_content(full)
    return response.text

def ask_groq(prompt: str, system: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=600,
        temperature=0.7,
    )
    return response.choices[0].message.content

@app.get("/")
def root():
    return {"status": "MarketMind API running ✅", "version": "1.0"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        system = SYSTEM_PROMPTS.get(req.mode, SYSTEM_PROMPTS["chat"])
        reply = ask_gemini(req.message, system)
        return {"reply": reply, "model": "gemini-1.5-flash", "mode": req.mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/campaign")
def generate_campaign(req: ChatRequest):
    try:
        reply = ask_groq(req.message, SYSTEM_PROMPTS["campaign"])
        return {"reply": reply, "model": "groq-llama3", "mode": "campaign"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pitch")
def generate_pitch(req: ChatRequest):
    try:
        reply = ask_gemini(req.message, SYSTEM_PROMPTS["pitch"])
        return {"reply": reply, "model": "gemini-1.5-flash", "mode": "pitch"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-lead")
def score_lead(req: LeadRequest):
    try:
        prompt = f"""Lead Details:
- Company: {req.company}
- Industry: {req.industry}
- Interactions this week: {req.interactions}
- Budget signal: {req.budget_signal}
Analyze and score this lead."""
        reply = ask_groq(prompt, SYSTEM_PROMPTS["leads"])
        return {"reply": reply, "model": "groq-llama3", "lead": req.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-market")
def analyze_market(req: ChatRequest):
    try:
        prompt = f"Provide a market analysis for: {req.message}. Include trends, opportunities, and competitor landscape."
        reply = ask_gemini(prompt, SYSTEM_PROMPTS["chat"])
        return {"reply": reply, "model": "gemini-1.5-flash", "mode": "market"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))