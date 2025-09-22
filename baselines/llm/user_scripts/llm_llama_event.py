import os
import csv
import json
import re
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "events_html"               # Thư mục chứa các file HTML sự kiện
CSV_FILE_LIST = "file_list3.csv"       # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_event3.csv"

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
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("Mô hình LLM đã tải xong.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("Đảm bảo đã cài 'transformers'/'torch', có GPU (hoặc đủ RAM) và đã có quyền truy cập Llama trên HF.")
    raise

# ==== LÀM SẠCH HTML (ưu tiên trang sự kiện) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Ưu tiên các block thường gặp của Event
    css_selectors = [
        "[itemtype*='schema.org/Event']",
        "[itemtype*='schema.org/MusicEvent']",
        "div.PnFP4htjx3zE9tlGdbDm, article#post-1208118, div.DefaultLayout__Container-sc-1k0r57y-0, div#app",
        "div#__next, div.event-main, section#details, div#details, div#event-details",
        "section.venue, div.venue, section.location, div.location, [itemprop='location']",
        "section.datetime, div.datetime, section#date, div.date-time, time",
        "section.performer, div.performer, section.artist, div.artist",
        "h1.event-title, h1[itemprop='name']",
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

# ==== CHUẨN HOÁ NGÀY/GIỜ ====
COMMON_DT_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%d/%m/%Y %H:%M", "%d/%m/%Y",
    "%d-%m-%Y %H:%M", "%d-%m-%Y",
    "%b %d, %Y %I:%M %p", "%b %d, %Y",
    "%B %d, %Y %I:%M %p", "%B %d, %Y",
]

def normalize_datetime(s: str) -> str:
    """Cố gắng chuẩn hoá về ISO 8601 (YYYY-MM-DDTHH:MM) hoặc trả nguyên nếu không parse được."""
    if not s: return ""
    t = re.sub(r"\s+", " ", s.strip())
    # Ưu tiên time/@datetime hoặc meta content ISO
    candidates = [t]
    # Thêm bản thay AM/PM viết thường
    if re.search(r"\b(am|pm)\b", t, re.I):
        candidates.append(re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), t, flags=re.I))

    for cand in candidates:
        for fmt in COMMON_DT_FORMATS:
            try:
                dt = datetime.strptime(cand, fmt)
                # Không timezone: in ISO local-naive
                return dt.strftime("%Y-%m-%dT%H:%M") if "%H" in fmt else dt.strftime("%Y-%m-%d")
            except Exception:
                continue

    # Bắt riêng date + time (ví dụ: "Aug 22, 2025 at 7:30 PM")
    m = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}).{0,10}(\d{1,2}:\d{2}\s*(AM|PM)?)", t, re.I)
    if m:
        try:
            d = datetime.strptime(m.group(1), "%B %d, %Y")
        except Exception:
            try:
                d = datetime.strptime(m.group(1), "%b %d, %Y")
            except Exception:
                d = None
        if d:
            tm = m.group(2).upper()
            try:
                tt = datetime.strptime(tm, "%I:%M %p").time()
            except Exception:
                try:
                    tt = datetime.strptime(tm, "%H:%M").time()
                except Exception:
                    tt = None
            if tt:
                return f"{d.strftime('%Y-%m-%d')}T{tt.strftime('%H:%M')}"

    # Không parse được → trả nguyên
    return t

# ==== FALLBACK (nếu LLM bỏ sót) ====
def fallback_extract_name(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("h1[itemprop='name'], h1.event-title")
    if node: return node.get_text(strip=True)
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"): return ogt["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_venue(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org location -> Place.name
    node = soup.select_one("[itemprop='location'] [itemprop='name'], [itemprop='location']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()
    # Text label
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Venue|Địa điểm|Location)[:\s]+([^\n]{2,120})", text, re.I)
    if m: return m.group(2).strip()
    # class/id common
    cand = soup.select_one(".venue, #venue, .location, #location, .place")
    if cand and cand.get_text(strip=True):
        return cand.get_text(" ", strip=True)
    return ""

def fallback_extract_artist(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org performer -> name
    nodes = soup.select("[itemprop='performer'] [itemprop='name'], [itemprop='performer'], [itemprop='actor'], [itemprop='artist']")
    values = []
    for n in nodes:
        val = n.get("content") or n.get_text()
        if val: values.append(val.strip())
    if values:
        # unique + join
        return ", ".join(dict.fromkeys([v for v in values if v]))
    # Text label
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Artist|Performer|Line[- ]?up|Nghệ sĩ)[:\s]+([^\n]{2,160})", text, re.I)
    if m: return m.group(2).strip()
    # class/id
    cand = soup.select_one(".artist, #artist, .performer, #performer, .lineup, #lineup")
    if cand and cand.get_text(strip=True):
        return cand.get_text(" ", strip=True)
    return ""

def fallback_extract_datetime(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org startDate
    for sel in ["[itemprop='startDate']", "meta[itemprop='startDate']", "time[datetime]"]:
        node = soup.select_one(sel)
        if node:
            val = node.get("content") or node.get("datetime") or node.get_text()
            if val:
                norm = normalize_datetime(val)
                if norm: return norm

    # Meta: og:start_time, event:start_time...
    node = soup.select_one("meta[property='event:start_time'], meta[name='event:start_time'], meta[property='og:start_time']")
    if node and node.get("content"):
        norm = normalize_datetime(node["content"])
        if norm: return norm

    # Text patterns
    text = soup.get_text(" ", strip=True)
    # YYYY-MM-DD HH:MM
    m = re.search(r"\b(20\d{2}-\d{1,2}-\d{1,2})\s+(\d{1,2}:\d{2}(?:\s*(AM|PM))?)", text, re.I)
    if m:
        return normalize_datetime(f"{m.group(1)} {m.group(2)}")
    # Month DD, YYYY [HH:MM AM/PM]
    m = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*20\d{2}(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)?)", text, re.I)
    if m:
        return normalize_datetime(m.group(1))
    # DD/MM/YYYY [HH:MM]
    m = re.search(r"(\d{1,2}/\d{1,2}/20\d{2}(?:\s+\d{1,2}:\d{2}(?:\s*(?:AM|PM))?)?)", text, re.I)
    if m:
        return normalize_datetime(m.group(1))
    return ""

# ==== TẠO PROMPT CHO LLAMA (domain EVENT) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # giới hạn để an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following event page text, extract these fields and return them in JSON format:\n"
        "- `name`: event title.\n"
        "- `venue`: event venue/place name.\n"
        "- `date_time`: event start date/time (prefer ISO like 'YYYY-MM-DDTHH:MM' if possible).\n"
        "- `artist`: performer/artist lineup; if multiple, join with comma.\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: name, venue, date_time, artist (all strings), "
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
    extracted = {"name":"", "venue":"", "date_time":"", "artist":""}
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

# ==== XỬ LÝ & GHI KẾT QUẢ ====
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["filename","name","venue","date_time","artist"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🎫 Đang xử lý các HTML sự kiện"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name":"","venue":"","date_time":"","artist":""})
            continue

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            clean_text = get_clean_text_from_html(html)
            prompt = create_llm_prompt_for_llama(clean_text, tokenizer)

            # Chỉ lấy phần sinh ra (không kèm prompt) để dễ parse
            llm_raw_response = pipe(prompt, return_full_text=False)[0]["generated_text"]
            extracted = parse_llm_response_to_json(llm_raw_response)

            # Fallback nếu LLM bỏ sót
            if not extracted.get("name"):
                extracted["name"] = fallback_extract_name(html)
            if not extracted.get("venue"):
                extracted["venue"] = fallback_extract_venue(html)
            if not extracted.get("artist"):
                extracted["artist"] = fallback_extract_artist(html)
            if not extracted.get("date_time"):
                extracted["date_time"] = fallback_extract_datetime(html)
            else:
                extracted["date_time"] = normalize_datetime(extracted["date_time"])

            extracted["filename"] = filename
            writer.writerow(extracted)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "name":"","venue":"","date_time":"","artist":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
