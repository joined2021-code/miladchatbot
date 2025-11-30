import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # <-- NEW
from fastapi.staticfiles import StaticFiles # <-- NEW
from pydantic import BaseModel # <-- NEW: برای تعریف مدل داده ورودی POST
from .orchestrator import get_reply_user

load_dotenv()

# تعریف مدل داده برای درخواست‌های POST به /reply
class UserMessage(BaseModel):
    user_message: str

app = FastAPI()

# اضافه کردن CORS (بدون تغییر)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# ۱. سرویس‌دهی روت اصلی (/) برای نمایش index.html
# ----------------------------------------------------------------------

# نصب StaticFiles برای سرویس‌دهی فایل‌های استاتیک فرانت‌اند (مثل CSS، JS، تصاویر)
# این خط فولدر 'frontend' را در مسیر /static_files در دسترس قرار می‌دهد
app.mount("/static_files", StaticFiles(directory="frontend"), name="frontend_static")

# روت اصلی که فایل index.html را برمی‌گرداند
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        # مسیر باید نسبت به جایی که uvicorn اجرا می‌شود، درست باشد
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # اگر فایل HTML پیدا نشد، یک خطای سرور بده
        raise HTTPException(status_code=500, detail="Frontend file not found.")

# ----------------------------------------------------------------------
# ۲. روت API برای دریافت پاسخ چت (تغییر یافته به POST)
# ----------------------------------------------------------------------

# روت API که با متد POST اجرا می‌شود
@app.post("/reply")
def reply(data: UserMessage):
    # دریافت متن پیام از مدل Pydantic
    user_text = data.user_message
    
    # گرفتن پاسخ از Gemini
    response_text = get_reply_user(user_text)
    
    # برگرداندن پاسخ
    return {"response": response_text}