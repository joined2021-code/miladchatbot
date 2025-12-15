import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .orchestrator import get_reply_user
# جایگزینی: openai به جای sambanova
from openai import OpenAI 

load_dotenv()

# --------------------------- مدل‌های ورودی ---------------------------
class UserMessage(BaseModel):
    user_message: str

class SummarizeRequest(BaseModel):
    text_to_summarize: str

# --------------------------- تنظیمات GapGPT ---------------------------
GAPGPT_API_KEY = os.getenv("GAPGPT_API_KEY")
GAPGPT_BASE_URL = os.getenv("GAPGPT_BASE_URL", "https://api.gapgpt.app/v1")
summary_model_client = None
SETUP_ERROR = None
# مدل grok-3-mini برای خلاصه‌سازی
MODEL_NAME = "grok-3-mini" 

if GAPGPT_API_KEY:
    try:
        # مقداردهی به کلاینت OpenAI با URL و کلید گپ جی‌پی‌تی
        summary_model_client = OpenAI(
            api_key=GAPGPT_API_KEY,
            base_url=GAPGPT_BASE_URL,
        )
        print("✅ GapGPT API initialized successfully for summary.")
    except Exception as e:
        print("⚠️ خطا در مقداردهی GapGPT:", e)
        SETUP_ERROR = str(e)
else:
    SETUP_ERROR = "GAPGPT_API_KEY یافت نشد."
    print("⚠️ ", SETUP_ERROR)

# --------------------------- اپ اصلی ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static_files", StaticFiles(directory="frontend"), name="frontend_static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Frontend file not found.")

# --------------------------- چت ---------------------------
@app.post("/reply")
async def reply(data: UserMessage):
    return {"response": get_reply_user(data.user_message)}

# --------------------------- خلاصه‌سازی ---------------------------
@app.post("/summarize")
async def summarize_text(data: SummarizeRequest):
    global summary_model_client, SETUP_ERROR
    
    if not summary_model_client or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=str(SETUP_ERROR))

    prompt = f"متن زیر را به فارسی و به صورت خلاصه توضیح بده:\n\n{data.text_to_summarize}"
    
    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        # فراخوانی API چت GapGPT
        resp = summary_model_client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages,
            temperature=0.1, 
        )
        
        if resp.choices and resp.choices[0].message:
            return {"summary": resp.choices[0].message.content.strip()}
        
        raise Exception("پاسخ نامعتبر از مدل دریافت شد.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در خلاصه‌سازی: {str(e)}")

# --------------------------- تست سلامت API ---------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "api_configured": GAPGPT_API_KEY is not None,
        "model": MODEL_NAME 
    }