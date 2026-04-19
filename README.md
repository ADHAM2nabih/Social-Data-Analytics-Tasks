# دليل تشغيل المشروع

هذا المستودع يحتوي أربع مهام رئيسية مرتبة في المجلدات `task1`, `task2`, `task3`, `task4`.
الملفات الناتجة تُخزن محليًا داخل `data/` و `final_data/` كما هو موضح في README لكل مهمة داخل كل مجلد.

**ملاحظة مهمة:** هذا الملف في جذر المشروع (خارج أي مجلد فرعي)، ويحتوي على جميع خطوات الإعداد والتشغيل المطلوبة.

## المتطلبات الأساسية
- Python 3.8+ (موصى به 3.10)
- Git (لنسخ المستودع إذا لزم)
- اتصال إنترنت (لـ API و pytrends وقراءة الأخبار)

## إنشاء بيئة افتراضية وتثبيت الحزم (مباشرة، بالأوامر)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r task1/requirements.txt -r task2/requirements.txt -r task3/requirements.txt
pip install -r task4/requirements.txt
```

Linux / macOS (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r task1/requirements.txt -r task2/requirements.txt -r task3/requirements.txt
```

يمكنك أيضاً تثبيت كل مجموعة تباعًا:

```bash
pip install -r task1/requirements.txt
pip install -r task2/requirements.txt
pip install -r task3/requirements.txt
pip install -r task4/requirements.txt
```

## متغيرات البيئة المطلوبة (Environment Variables)
ضع هذه القيم في ملف `.env` في جذر المشروع أو عيّنها في الشل قبل التشغيل.

قائمة المتغيرات الأساسية:

- `REDDIT_CLIENT_ID` — معرف تطبيق Reddit (مطلوب لـ PRAW)
- `REDDIT_CLIENT_SECRET` — سر تطبيق Reddit (مطلوب)
- `NEWS_API_KEY` — مفتاح NewsAPI (مطلوب لبحث الأخبار)
- `REDDIT_USER_AGENT` — اختياري (يوجد قيمة افتراضية `social_analytics_project/1.0`)
- `GEMINI_API_KEY` — اختياري، مطلوب فقط إذا أردت تشغيل تسميات أو استدعاءات LLM في Task 3
- `GLOVE_PATH` — مسار اختياري لملف GloVe إن استخدمت تمثيلات كلمات محلية

مثال محتوى ملف `.env`:

```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
NEWS_API_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_api_key
GLOVE_PATH=
REDDIT_USER_AGENT=social_analytics_project/1.0
```

بدائل لضبط المتغيرات مؤقتًا في الشل:

PowerShell:

```powershell
$env:REDDIT_CLIENT_ID = "..."
$env:REDDIT_CLIENT_SECRET = "..."
$env:NEWS_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
```

Bash:

```bash
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
export NEWS_API_KEY="..."
export GEMINI_API_KEY="..."
```

## تشغيل المهام (Commands)

ملاحظة: يمكنك تشغيل السكربتات من جذر المستودع أو من داخل مجلد كل مهمة.

- Task 1 — جمع البيانات وتحليل الاتجاهات/ريديت/أخبار:

```bash
python task1/main.py --topn 10 --reddit_limit 100 --news_per_term 3 --geo GLOBAL
```

المخرجات: ملفات CSV تُنشأ داخل مجلد `data/` (مثل `trends_<run_id>.csv`, `news_<run_id>.csv`, `reddit_posts_enriched_<run_id>.csv`) وتُدمج تراكمياً في `final_data/`.

- Task 2 — المعالجة المسبقة للنصوص:

```bash
python task2/main.py --top_k 200
```

المدخل الافتراضي: `..\\task1\\final_data\\reddit_posts_enriched.csv` (انظر `task2/README.md`). مخرجات المعالجة تحفظ في `task2/final_data/processed/`.

- Task 3 — بناء نموذج الانفعال الكامل وتوليد الملصقات:

```bash
python task3/main.py --sample_size 200 --gemini_api_key YOUR_KEY
```

بدلاً من تمرير المفتاح في السطر، يمكنك تعيين `GEMINI_API_KEY` في البيئة قبل التشغيل.
المخرجات تُخزن تحت `task3/final_data/` (labels, models, reports، الخ).

- Task 4 — تقييم وتحسين النماذج + Error Analysis + نشر API:

```bash
python task4/main.py
```

لتشغيل API بعد التدريب:

```bash
cd task4
uvicorn api:app --reload --port 8000
```

## نقاط تحرّي الخلل السريعة
- إذا ظهر `NEWS_API_KEY not set` أو `Reddit credentials missing.`: تأكد من أن `.env` موجود أو أنك ضبطت المتغيرات في الشل.
- pytrends قد يفشل أحياناً بسبب قيود Google Trends; الكود سيحاول بدائل (NewsAPI أو قائمة افتراضية).
- إن كان هناك تحذير عن `REQUESTS_CA_BUNDLE` أو `SSL_CERT_FILE` فإن `task1/utils/config.py` سيحاول إلغاء ضبط المتغيرات إذا كانت تشير لملف غير موجود.

## مراجع داخلية
- تعليمات ومعلومات إضافية لكل مهمة موجودة في:
  - [task1 README](task1/README.md)
  - [task2 README](task2/README.md)
  - [task3 README](task3/README.md)

## إن احتجت مساعدة إضافية
- أستطيع: إضافة ملف `.env.example` تلقائي، تشغيل اختبارات محلية أو عمل commit و push لملف README. أخبرني أي خطوة تريدني أن أنفذها.






Set-Location "c:\Users\Rowan\OneDrive\Desktop\collage\semester8\social analytics\tasks\Social-Data-Analytics-Tasks\task4"; $env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"; & "c:/Users/Rowan/OneDrive/Desktop/collage/semester8/social analytics/tasks/.venv/Scripts/python.exe" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501