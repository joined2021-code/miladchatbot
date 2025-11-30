import os
from dotenv import load_dotenv
import google.generativeai as genai
# نیازی به وارد کردن این موارد نیست مگر برای تنظیمات پیشرفته
# from google.generativeai.types import HarmCategory, HarmBlockThreshold 

load_dotenv()  # بارگذاری متغیرهای محیطی از فایل .env

# بررسی کلید API به صورت سراسری
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = None # تعریف اولیه مدل

if not GEMINI_API_KEY:
    print("⚠️ خطا: متغیر محیطی GEMINI_API_KEY تنظیم نشده است!")
else:
    try:
        # تنظیم کلاینت Gemini
        genai.configure(api_key=GEMINI_API_KEY)

        # ساخت مدل با پرامپت سیستم
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction="تو یک دستیار مفید و باهوش هستی که به زبان فارسی پاسخ می‌دی و سوالات دانشجویی پاسخ میدی."
        )
    except Exception as e:
        print(f"⚠️ خطا در تنظیم Gemini API: {str(e)}")
        # در صورت شکست، مدل همچنان None خواهد بود

def get_reply_user(user_text: str) -> str:
    """
    این تابع متن کاربر رو می‌گیره و از Google Gemini پاسخ واقعی می‌گیره.
    """
    # بررسی مدل قبل از استفاده (برای مدیریت خطای API Key)
    if model is None:
        return "⚠️ خطای تنظیمات بک‌اند: کلید API (GEMINI_API_KEY) احتمالاً تنظیم نشده یا نامعتبر است. لطفاً فایل‌های پیکربندی را بررسی کنید."

    try:
        # استفاده از ابزار جستجوی گوگل برای پاسخ‌های به‌روز (با tools=[{"google_search": {}}])
        response = model.generate_content(
            user_text,
            tools=[{"google_search": {}}]
        )
        
        # بررسی محتوای پاسخ
        if response.text:
            return response.text.strip()
        else:
            return "پاسخی از مدل دریافت نشد. شاید به دلیل محتوای نامناسب مسدود شده باشد."

    except Exception as e:
        # خطای زمان اجرای API
        return f"خطا در گرفتن پاسخ از Gemini: {str(e)}"