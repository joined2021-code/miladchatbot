import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .orchestrator import get_reply_user # این تابع برای پاسخگویی اصلی چت است
import base64
import io
import struct

# --- وارد کردن ابزارهای Gemini API ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

# تعریف مدل داده برای درخواست‌های POST
class UserMessage(BaseModel):
    user_message: str

# تعریف مدل داده برای درخواست‌های TTS
class TTSRequest(BaseModel):
    text: str
    voice: str = "Kore" # صدای پیش‌فرض مناسب فارسی

# تعریف مدل داده برای درخواست خلاصه‌سازی
class SummarizeRequest(BaseModel):
    text_to_summarize: str

# --- متغیرهای جهانی و تنظیمات API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
tts_model = None
chat_model = None # برای استفاده در خلاصه‌سازی
SETUP_ERROR = None

if GEMINI_API_KEY:
    try:
        # تنظیم کلاینت Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        # مدل TTS و چت (برای خلاصه‌سازی) را مستقیماً تعریف می‌کنیم
        tts_model = genai.GenerativeModel('gemini-2.5-flash-preview-tts')
        chat_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"⚠️ خطا در تنظیم Gemini API: {str(e)}")
        SETUP_ERROR = f"خطا در تنظیمات اولیه مدل: {str(e)}"
else:
    SETUP_ERROR = "کلید API (GEMINI_API_KEY) در فایل‌های محیطی (مثل .env) یافت نشد."

# ----------------------------------------------------------------------
# توابع کمکی تبدیل PCM به WAV
# ----------------------------------------------------------------------

def pcm_to_wav(pcm_data: bytes, sample_rate=24000):
    """
    داده خام PCM را به فرمت فایل WAV تبدیل می‌کند (24kHz, 16-bit, Mono).
    """
    wav_file = io.BytesIO()
    num_channels = 1
    sample_width = 2
    byte_rate = sample_rate * num_channels * sample_width
    data_size = len(pcm_data)
    
    # Write WAV Header (RIFF Chunk)
    wav_file.write(b'RIFF') 
    wav_file.write(struct.pack('<I', 36 + data_size)) 
    wav_file.write(b'WAVE') 

    # Write FMT Sub-chunk
    wav_file.write(b'fmt ') 
    wav_file.write(struct.pack('<I', 16))
    wav_file.write(struct.pack('<H', 1))
    wav_file.write(struct.pack('<H', num_channels))
    wav_file.write(struct.pack('<I', sample_rate))
    wav_file.write(struct.pack('<I', byte_rate))
    wav_file.write(struct.pack('<H', num_channels * sample_width))
    wav_file.write(struct.pack('<H', sample_width * 8))

    # Write DATA Sub-chunk
    wav_file.write(b'data') 
    wav_file.write(struct.pack('<I', data_size))
    wav_file.write(pcm_data) 

    wav_file.seek(0)
    return wav_file.read()

# ----------------------------------------------------------------------
# سرویس‌های FastAPI
# ----------------------------------------------------------------------

app = FastAPI()

# اضافه کردن CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ۱. سرویس‌دهی روت اصلی (/) برای نمایش index.html
app.mount("/static_files", StaticFiles(directory="frontend"), name="frontend_static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Frontend file not found.")

# ۲. روت API برای دریافت پاسخ چت
@app.post("/reply")
def reply(data: UserMessage):
    # این روت از orchestrator.py استفاده می‌کند
    user_text = data.user_message
    response_text = get_reply_user(user_text)
    return {"response": response_text}

# ۳. روت API جدید برای TTS (تبدیل متن به گفتار)
@app.post("/tts")
async def generate_tts(data: TTSRequest):
    if tts_model is None or SETUP_ERROR:
        # اگر تنظیمات شکست خورده است، خطا برمی‌گردانیم
        raise HTTPException(status_code=500, detail=f"TTS model setup failed: {SETUP_ERROR}")

    # محدودیت TTS: 300 کاراکتر
    text_to_speak = data.text[:300]

    tts_payload = {
        "contents": [{
            "parts": [{ "text": text_to_speak }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { 
                        "voiceName": data.voice 
                    }
                }
            }
        }
    }

    try:
        response = tts_model.generate_content(**tts_payload, tools=[])
        
        audio_part = response.candidates[0].content.parts[0].inlineData
        
        if not audio_part or audio_part.mimeType != "audio/L16;rate=24000":
            raise HTTPException(status_code=500, detail="Invalid audio response from TTS model.")

        pcm_data_base64 = audio_part.data
        pcm_data_bytes = base64.b64decode(pcm_data_base64)
        wav_bytes = pcm_to_wav(pcm_data_bytes, sample_rate=24000)

        # ارسال فایل WAV به صورت بیس64 در JSON
        return JSONResponse(
            content={"audio_data": base64.b64encode(wav_bytes).decode('utf-8')},
            media_type="application/json"
        )

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")


# ۴. روت API جدید برای خلاصه‌سازی متن
@app.post("/summarize")
async def summarize_text(data: SummarizeRequest):
    if chat_model is None or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=f"Summarization model setup failed: {SETUP_ERROR}")

    text_to_summarize = data.text_to_summarize
    
    # ساخت پرامپت خلاصه‌سازی
    summary_prompt = (
        "متن زیر را به صورت مختصر و در حد یک پاراگراف، به زبان فارسی خلاصه کن:\n\n"
        f"متن: \"{text_to_summarize}\""
    )

    try:
        response = chat_model.generate_content(summary_prompt, tools=[])
        
        if response.candidates and response.candidates[0].finish_reason.name == 'SAFETY':
             return JSONResponse(
                 content={"summary": "⚠️ به دلیل خط‌مشی‌های ایمنی، امکان خلاصه‌سازی این متن وجود ندارد."},
                 media_type="application/json"
             )
        
        summary_text = response.text.strip()
        return JSONResponse(
            content={"summary": summary_text},
            media_type="application/json"
        )
        
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")