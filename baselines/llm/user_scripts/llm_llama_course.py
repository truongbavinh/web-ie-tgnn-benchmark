import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "course_html"              # Thư mục chứa các file HTML khóa học
CSV_FILE_LIST = "file_list3.csv"       # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_course3.csv"

# Dùng Llama 3.1 Instruct 8B
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ==== TẢI MÔ HÌNH ====
print(f"Đang tải mô hình LLM: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Cố dùng flash-attn nếu có; nếu không thì fallback tự động
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, attn_implementation="flash_attention_2", **model_kwargs
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        # temperature=0.1, # Không cần thiết khi do_sample=False
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("Mô hình LLM đã tải xong.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("Đảm bảo đã cài 'transformers'/'torch', có GPU (hoặc đủ RAM) và có quyền truy cập Llama trên HF.")
    raise

# ==== LÀM SẠCH HTML (ưu tiên trang Course) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Ưu tiên các block thường gặp: schema.org Course, syllabus/overview/price/instructor
    css_selectors = [
        "[itemtype*='schema.org/Course']",
        "div.css-9t6por", # Phổ biến cho trang chi tiết khóa học
        "div.flex",
        "main#main-content",
        "main#main-content-anchor",
        "div.main-content",
        "article.course-page",
        "div#course-main-content",
        "div.page-content", # Ví dụ từ cấu trúc chia cột
        "div.wrapper", # Bộ chọn chung, đôi khi hữu ích
        "div.container-fluid.ContentWrapper", # Bộ chọn chung khác
        "div[data-component='course-details']", # Các trang dùng framework
        "div.css-xxxxxx", # Các lớp CSS tự động sinh ra có thể cần phân tích
        "div.instructor-details", # Thông tin giảng viên
        "div.course-fees", # Thông tin học phí
        "div.course-duration" # Thông tin thời lượng
    ]

    main_content = None
    for sel in css_selectors:
        main_content = soup.select_one(sel)
        if main_content:
            break

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body")
        text = (body or soup).get_text(separator="\n", strip=True)

    text = "\n".join(filter(lambda x: x.strip(), text.split("\n")))
    return text

# ==== CHUẨN HOÁ & TIỆN ÍCH (duration, fees) ====
def minutes_from_iso8601(duration: str) -> int:
    """
    Parse ISO8601: PnW/PnD/TnHnMnS, PTnHnM..., trả phút; -1 nếu không parse được.
    """
    if not duration: return -1
    t = duration.strip().upper()

    # PnW (tuần), PnD (ngày)
    m = re.fullmatch(r"P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", t)
    if not m:
        return -1
    w = int(m.group(1) or 0)
    d = int(m.group(2) or 0)
    h = int(m.group(3) or 0)
    mn = int(m.group(4) or 0)
    s = int(m.group(5) or 0)
    total_min = w*7*24*60 + d*24*60 + h*60 + mn + s//60
    return total_min if total_min > 0 else -1

def minutes_from_text(s: str) -> int:
    """
    Parse thời lượng dạng text: '6 weeks', '10 hours', '1h30m', '45 minutes', '3 days', etc. -> phút.
    Ưu tiên trả về phút; nếu chỉ tuần/tháng không chắc đổi chính xác thì sẽ trả -1 (để in nguyên).
    """
    if not s: return -1
    t = s.lower().strip()

    # hh:mm
    m = re.search(r"\b(\d{1,2}):(\d{1,2})\b", t)
    if m:
        return int(m.group(1))*60 + int(m.group(2))

    # X hours + Y minutes
    h = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours|giờ|gio)", t)
    mn = re.search(r"(\d+)\s*(m|min|mins|minute|minutes|phút|phut|p)\b", t)
    if h and mn:
        return int(h.group(1))*60 + int(mn.group(1))
    if h:
        return int(h.group(1))*60
    if mn:
        return int(mn.group(1))

    # 1h30m, 2h05
    m = re.search(r"(\d+)\s*h\s*(\d{1,2})\s*m?", t)
    if m:
        return int(m.group(1))*60 + int(m.group(2))

    # days (quy đổi 1 day = 8h học? Không chuẩn; để nguyên -> trả -1)
    if re.search(r"\b\d+\s*(day|days|ngày)\b", t):
        return -1
    # weeks / months -> để nguyên
    if re.search(r"\b\d+\s*(week|weeks|tuần|month|months|tháng)\b", t):
        return -1

    return -1

def fmt_duration(raw_text: str) -> str:
    """
    Chuẩn hoá duration: Nếu parse được phút -> 'Hh Mm' (ví dụ '1h 30m' hoặc '45m').
    Nếu phát hiện weeks/days/months -> trả nguyên cho dễ hiểu ('6 weeks', '3 months').
    """
    if not raw_text: return ""
    # Ưu tiên ISO8601
    mins = minutes_from_iso8601(raw_text)
    if mins <= 0:
        # Thử text tự do
        mins = minutes_from_text(raw_text)

    # Nếu chứa weeks/days/months → để nguyên
    if re.search(r"\b\d+\s*(week|weeks|tuần|day|days|ngày|month|months|tháng)\b", raw_text.lower()):
        return re.sub(r"\s+", " ", raw_text.strip())

    if mins > 0:
        h, m = divmod(mins, 60)
        return (f"{h}h {m}m" if h else f"{m}m").strip()

    # Không parse được → trả nguyên
    return re.sub(r"\s+", " ", raw_text.strip())

def parse_fees_text(text: str) -> str:
    """
    Bắt giá tiền: hỗ trợ USD/VND/€… & 'Free/Miễn phí'.
    """
    if not text: return ""
    t = " ".join(text.split())
    # Free
    m = re.search(r"\b(Free|Miễn phí|Miens phi)\b", t, re.I)
    if m: return m.group(1).title()

    # Tiền tệ
    m = re.search(r"((?:USD|US\$|\$|€|£|¥|₩|₫|VND|AUD|CAD|JPY|NT\$|SGD|HKD)\s?\d[\d., ]+)", t, re.I)
    if m: return m.group(1).strip()

    # Label Fees/Tuition/Price
    m = re.search(r"(Tuition|Học phí|Fees?|Price|Giá)\s*[:\-]?\s*([^|,\n]{1,60})", t, re.I)
    if m: return f"{m.group(1)}: {m.group(2).strip()}"
    return ""

# ==== FALLBACK (nếu LLM bỏ sót) ====
def fallback_extract_title(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("h1[itemprop='name'], h1.course-title")
    if node: return node.get_text(strip=True)
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"): return ogt["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_subject(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org about (Course.about)
    node = soup.select_one("[itemprop='about'], meta[itemprop='about']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()

    # Breadcrumbs / category
    cat = soup.select_one(".breadcrumb, nav.breadcrumb, .category, .subject, [data-testid='subject']")
    if cat and cat.get_text(strip=True):
        return cat.get_text(" / ", strip=True)

    # meta keywords
    mk = soup.select_one("meta[name='keywords']")
    if mk and mk.get("content"):
        return mk["content"].strip()

    # Label 'Subject: ...' / 'Chủ đề: ...'
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Subject|Chủ đề|Ngành)[:\s]+([^\n]{2,80})", text, re.I)
    if m: return m.group(2).strip()
    return ""

def fallback_extract_duration(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org duration / timeRequired
    node = soup.select_one("[itemprop='duration'], meta[itemprop='duration'], [itemprop='timeRequired'], meta[itemprop='timeRequired']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return fmt_duration(val)

    # Labels: Duration/Length/Thời lượng
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Duration|Length|Thời\s*lượng)[:\s]+([^\n]{1,60})", text, re.I)
    if m:
        return fmt_duration(m.group(2))

    # Các pattern phổ biến trên landing: 'X hours', 'Y weeks', 'Self-paced'
    m = re.search(r"(\d+\s*(hours?|hrs?|h|minutes?|mins?|m|weeks?|months?|days?))", text, re.I)
    if m:
        return fmt_duration(m.group(1))
    m = re.search(r"\b(Self[-\s]?paced|Tự học theo nhịp độ)\b", text, re.I)
    if m:
        return m.group(1)
    return ""

def fallback_extract_fees(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org price / offers
    p = soup.select_one("[itemprop='price'], meta[itemprop='price']")
    if p:
        val = p.get("content") or p.get_text()
        cur = ""
        cnode = soup.select_one("[itemprop='priceCurrency'], meta[itemprop='priceCurrency']")
        if cnode:
            cur = (cnode.get("content") or cnode.get_text() or "").strip()
        if val:
            return f"{cur} {val}".strip()
    # Text
    return parse_fees_text(soup.get_text(" ", strip=True))

def fallback_extract_instructor(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org instructor/teacher/creator
    for sel in ["[itemprop='instructor'] [itemprop='name']",
                "[itemprop='instructor']",
                "[itemprop='teacher'] [itemprop='name']",
                "[itemprop='teacher']",
                "[itemprop='creator'] [itemprop='name']",
                "meta[name='author']"]:
        node = soup.select_one(sel)
        if node:
            val = node.get("content") or node.get_text()
            if val: return val.strip()

    # Khối instructor/faculty
    blk = soup.select_one(".instructor, .faculty, #instructor, #faculty")
    if blk and blk.get_text(strip=True):
        return blk.get_text(" ", strip=True)

    # "Instructor: ..." / "Giảng viên: ..."
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Instructor|Giảng viên|Giang vien|Lecturer|Giáo viên)[:\s]+([^\n]{3,80})", text, re.I)
    if m: return m.group(2).strip()
    return ""

# ==== TẠO PROMPT CHO LLAMA (domain COURSE) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # giới hạn để an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following course page text, extract these fields and return them in JSON format:\n"
        "- `title`: official course title.\n"
        "- `subject`: subject/discipline/category of the course.\n"
        "- `duration`: course length (e.g., '10h', '6 weeks', 'Self-paced'). Use a compact, human-readable form.\n"
        "- `fees`: tuition or price; include currency if present, or 'Free' when applicable.\n"
        "- `instructor`: main instructor/lecturer name(s).\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: title, subject, duration, fees, instructor (all strings), "
        "and wrap it like:\n```json\n{...}\n```\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_str

# ==== PARSE JSON TỪ ĐẦU RA ====
def parse_llm_response_to_json(llm_response):
    extracted = {"title":"", "subject":"","duration":"","fees":"","instructor":""}
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            s, e = llm_response.find("{"), llm_response.rfind("}")
            if s != -1 and e != -1 and e > s:
                json_str = llm_response[s:e+1]
            else:
                raise ValueError("No valid JSON structure found in LLM response.")
        data = json.loads(json_str)

        def safe_str(v):
            if isinstance(v, list): return ", ".join(map(str, v)).strip()
            if v is None: return ""
            return str(v).strip()

        for k in list(extracted.keys()):
            extracted[k] = safe_str(data.get(k))
    except Exception as e:
        print(f"⚠️ Lỗi phân tích JSON: {e}")
        print(f"LLM Response (full): {llm_response}")
    return extracted

# ==== ĐỌC DANH SÁCH FILE ====
try:
    with open(CSV_FILE_LIST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filenames = [row["filename"] for row in reader]
except FileNotFoundError:
    print(f"Lỗi: Không thấy tệp '{CSV_FILE_LIST}'.")
    raise

# =======================================================
# ==== [PHẦN BỔ SUNG] XỬ LÝ & GHI KẾT QUẢ ====
# =======================================================

all_results = []
print(f"\nBắt đầu xử lý {len(filenames)} file...")

for filename in tqdm(filenames, desc="Processing HTML files"):
    html_path = os.path.join(HTML_DIR, filename)
    
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"⚠️ Cảnh báo: Bỏ qua file không tồn tại: {html_path}")
        continue
    except Exception as e:
        print(f"⚠️ Cảnh báo: Lỗi khi đọc file {html_path}: {e}")
        continue

    # 1. Làm sạch HTML và tạo prompt
    clean_text = get_clean_text_from_html(html_content)
    prompt = create_llm_prompt_for_llama(clean_text, tokenizer)

    # 2. Chạy LLM pipeline
    try:
        response_full = pipe(prompt)[0]["generated_text"]
        # Chỉ lấy phần text do LLM tạo ra, bỏ qua phần prompt
        llm_output_text = response_full.split("<|end_header_id|>")[-1].strip()
        
        # 3. Phân tích kết quả từ LLM
        extracted_data = parse_llm_response_to_json(llm_output_text)
        extracted_data['filename'] = filename # Thêm tên file để tham chiếu

    except Exception as e:
        print(f"Lỗi khi chạy pipeline cho file {filename}: {e}")
        # Khởi tạo dữ liệu rỗng để chạy fallback
        extracted_data = {"filename": filename, "title":"", "subject":"","duration":"","fees":"","instructor":""}

    # 4. Fallback: Nếu LLM bỏ sót trường nào, dùng hàm trích xuất dự phòng
    if not extracted_data.get("title"):
        extracted_data["title"] = fallback_extract_title(html_content)
    if not extracted_data.get("subject"):
        extracted_data["subject"] = fallback_extract_subject(html_content)
    if not extracted_data.get("duration"):
        # Chuẩn hoá duration sau khi fallback
        raw_duration = fallback_extract_duration(html_content)
        extracted_data["duration"] = fmt_duration(raw_duration) if raw_duration else ""
    if not extracted_data.get("fees"):
        extracted_data["fees"] = fallback_extract_fees(html_content)
    if not extracted_data.get("instructor"):
        extracted_data["instructor"] = fallback_extract_instructor(html_content)

    all_results.append(extracted_data)

# 5. Ghi tất cả kết quả ra file CSV
if all_results:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["filename", "title", "subject", "duration", "fees", "instructor"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✅ Xong! Kết quả đã được ghi vào file '{OUTPUT_CSV}'")
else:
    print("\n⚠️ Không có file nào được xử lý hoặc không có kết quả để ghi.")