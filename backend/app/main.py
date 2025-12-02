# ---------------------------- main.py (نسخه‌ی اصلاح‌شده کامل) ----------------------------
# این نسخه شامل اصلاح کامل TTS، رفع خطاهای generationConfig، و سازگاری با Gemini API جدید است.

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .orchestrator import get_reply_user
import base64
import io
import struct
import google.generativeai as genai

load_dotenv()

# --------------------------- مدل‌های ورودی ---------------------------

class UserMessage(BaseModel):
    user_message: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "Kore"

class SummarizeRequest(BaseModel):
    text_to_summarize: str

# --------------------------- تنظیم Gemini ---------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
tts_model_client = None
chat_model_client = None
SETUP_ERROR = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        tts_model_client = genai.GenerativeModel('gemini-2.5-flash-preview-tts')
        chat_model_client = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini API initialized.")
    except Exception as e:
        print("⚠️ خطا در مقداردهی Gemini:", e)
        SETUP_ERROR = str(e)
else:
    SETUP_ERROR = "GEMINI_API_KEY یافت نشد."
    print("⚠️ ", SETUP_ERROR)

# --------------------------- تبدیل PCM به WAV ---------------------------

def pcm_to_wav(pcm_data: bytes, sample_rate=24000):
    wav_file = io.BytesIO()
    num_channels = 1
    sample_width = 2
    byte_rate = sample_rate * num_channels * sample_width
    data_size = len(pcm_data)

    wav_file.write(b'RIFF')
    wav_file.write(struct.pack('<I', 36 + data_size))
    wav_file.write(b'WAVE')

    wav_file.write(b'fmt ')
    wav_file.write(struct.pack('<I', 16))
    wav_file.write(struct.pack('<H', 1))
    wav_file.write(struct.pack('<H', num_channels))
    wav_file.write(struct.pack('<I', sample_rate))
    wav_file.write(struct.pack('<I', byte_rate))
    wav_file.write(struct.pack('<H', num_channels * sample_width))
    wav_file.write(struct.pack('<H', sample_width * 8))

    wav_file.write(b'data')
    wav_file.write(struct.pack('<I', data_size))
    wav_file.write(pcm_data)

    wav_file.seek(0)
    return wav_file.read()

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
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# --------------------------- چت ---------------------------

@app.post("/reply")
def reply(data: UserMessage):
    return {"response": get_reply_user(data.user_message)}

# --------------------------- TTS ---------------------------

@app.post("/tts")
async def generate_tts(data: TTSRequest):
    global tts_model_client, SETUP_ERROR

    if not tts_model_client or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=str(SETUP_ERROR))

    text_to_speak = data.text[:300]

    try:
        response = tts_model_client.generate_content(
            contents=[{
                "parts": [{"text": text_to_speak}]
            }],
            generation_config={
                "response_modalities": ["AUDIO"],
                "audio_config": {
                    "voice_name": data.voice,
                    "audio_format": "wav"
                }
            }
        )

        audio_part = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                audio_part = part.inline_data
                break

        if not audio_part:
            raise RuntimeError("Could not extract inline audio data")

        if audio_part.mime_type != "audio/wav":
            raise RuntimeError(f"Unexpected MIME type: {audio_part.mime_type}")

        wav_bytes = base64.b64decode(audio_part.data)

        return {"audio_data": base64.b64encode(wav_bytes).decode("utf-8")}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

# --------------------------- خلاصه‌سازی ---------------------------

@app.post("/summarize")
async def summarize_text(data: SummarizeRequest):
    global chat_model_client, SETUP_ERROR

    if not chat_model_client or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=str(SETUP_ERROR))

    prompt = (
        "متن زیر را کوتاه و خلاصه کن:\n\n"
        f"{data.text_to_summarize}"
    )

    try:
        resp = chat_model_client.generate_content(prompt, tools=[])
        return {"summary": resp.text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
