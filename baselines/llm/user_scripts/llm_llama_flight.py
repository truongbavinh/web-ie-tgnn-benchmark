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
HTML_DIR = "flights_html"            # Thư mục chứa các file HTML chuyến bay
CSV_FILE_LIST = "file_list3.csv"      # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_flights3.csv"

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

# ==== LÀM SẠCH HTML (ưu tiên trang flight/itinerary/pricing) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Khối thường gặp: schema.org Flight/Reservation/Offer; itinerary/segment/price
    css_selectors = [
        "[itemtype*='schema.org/Flight']",
        "[itemtype*='schema.org/FlightReservation']",
        "div.main.clearfix", # generic, but sometimes useful
        "div.FlightItineraryDetails__details__fc", # from some flight sites
        "div.LoWCY", # generic content wrapper
        "main", # HTML5 main content tag
        "div.OQa--content", # generic content wrapper
        "div.ReactModal__Content", # often contains pop-up details
        "div#root", # common root element for React/SPA apps
        "div.results-summary",
        "div.results-page",
        "div.flight-details",
        "section.flight-info",
        "div.booking-details",
        "div.trip-details",
        "div.fare-details",
        "div.flight-card", # common for individual flight listings
        "div.booking-detail-item",
        "div.itinerary-section",
        "div.content-wrapper",
        "div.flight-info-container",
        "div.booking-details-wrapper", # Specific to some booking sites
        "div.details-segment", # Often contains segment details
        "div.flight-summary", # Summaries
        "div[data-component='flight-results-page']", # Kayak specific
        "div.section.module.itinerary", # TripAdvisor
        "div.section.module.flight-options", # TripAdvisor
        "div.flight-result-details", # General result details
        "div.leg-details", # For individual flight legs
        "div.flex-container", # Sometimes used for layout of flight info
        "div.travel-information"
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

# ==== CHUẨN HOÁ & TIỆN ÍCH (datetime, duration, price) ====
COMMON_DT_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%d/%m/%Y %H:%M", "%d/%m/%Y",
    "%d-%m-%Y %H:%M", "%d-%m-%Y",
    "%b %d, %Y %I:%M %p", "%b %d, %Y",
    "%B %d, %Y %I:%M %p", "%B %d, %Y",
]

def normalize_datetime(s: str) -> str:
    """Ưu tiên trả ISO 'YYYY-MM-DDTHH:MM' (nếu parse được); không thì trả nguyên (đã rút gọn khoảng trắng)."""
    if not s: return ""
    t = re.sub(r"\s+", " ", s.strip())
    cands = [t]
    if re.search(r"\b(am|pm)\b", t, re.I):
        cands.append(re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), t, flags=re.I))
    for cand in cands:
        for fmt in COMMON_DT_FORMATS:
            try:
                dt = datetime.strptime(cand, fmt)
                return dt.strftime("%Y-%m-%dT%H:%M") if "%H" in fmt else dt.strftime("%Y-%m-%d")
            except Exception:
                continue
    # HH:MM (không có ngày)
    m = re.search(r"\b(\d{1,2}:\d{2}(?:\s*(AM|PM))?)\b", t, re.I)
    if m:
        hhmm = m.group(1).upper()
        try:
            return datetime.strptime(hhmm, "%I:%M %p").strftime("%H:%M")
        except:
            try:
                return datetime.strptime(hhmm, "%H:%M").strftime("%H:%M")
            except:
                pass
    return t

def fmt_duration_text(s: str) -> str:
    """Đưa duration về dạng gọn 'Xh Ym' nếu bắt được (ví dụ '2h 15m', '1h', '45m')."""
    if not s: return ""
    t = s.lower()
    # hh:mm
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return (f"{h}h {mn}m" if h else f"{mn}m").strip()
    # X hours Y minutes / 2h 15m / 2 hr / 135 minutes
    h = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)", t)
    mn = re.search(r"(\d+)\s*(m|min|mins|minute|minutes)", t)
    if h and mn:
        return f"{int(h.group(1))}h {int(mn.group(1))}m"
    if h:
        return f"{int(h.group(1))}h"
    if mn:
        return f"{int(mn.group(1))}m"
    # “Duration: 2h15”/“2h15m”
    m = re.search(r"(\d+)\s*h\s*(\d{1,2})\s*m?", t)
    if m:
        return f"{int(m.group(1))}h {int(m.group(2))}m"
    return re.sub(r"\s+", " ", s.strip())

def parse_price(text: str) -> str:
    if not text: return ""
    t = " ".join(text.split())
    # Free hiếm trong flight, nhưng cứ hỗ trợ
    m = re.search(r"\b(Free|Miễn phí)\b", t, re.I)
    if m: return m.group(1).title()
    m = re.search(r"((?:USD|US\$|\$|€|£|¥|₩|₫|VND|AUD|CAD|JPY|NT\$|SGD|HKD)\s?\d[\d., ]+)", t, re.I)
    if m: return m.group(1).strip()
    m = re.search(r"(Price|Giá|Fare|Total)\s*[:\-]?\s*([^|,\n]{1,60})", t, re.I)
    if m: return f"{m.group(1)}: {m.group(2).strip()}"
    return ""

# ==== FALLBACK: lấy trực tiếp từ HTML nếu LLM bỏ sót ====
AIRLINE_HINTS = [
    "Vietnam Airlines","VietJet","VietJet Air","Bamboo Airways","Pacific Airlines",
    "AirAsia","Thai AirAsia","Singapore Airlines","Scoot","Jetstar",
    "Qatar Airways","Emirates","Etihad","Cathay Pacific","HK Express",
    "ANA","JAL","Korean Air","Asiana","China Airlines","EVA Air",
    "Turkish Airlines","Malaysia Airlines","Garuda Indonesia","Philippine Airlines",
    "Qantas","United Airlines","American Airlines","Delta Air Lines","Alaska Airlines",
    "British Airways","Lufthansa","Swiss","KLM","Air France"
]

def fallback_extract_name(html):
    soup = BeautifulSoup(html, "html.parser")
    # flight number pattern: 2-3 letters/digits + 2-4 digits (VN123, SQ321, VJ 640)
    text = soup.get_text(" ", strip=True)
    m = re.search(r"\b([A-Z0-9]{2,3})\s?-?\s?(\d{2,4})\b", text)
    flight_no = f"{m.group(1)} {m.group(2)}" if m else ""
    # airline (see fallback_extract_airline)
    airline = fallback_extract_airline(html)
    if airline and flight_no:
        return f"{airline} {flight_no}"
    if flight_no:
        return flight_no
    # og:title / h1
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"):
        return ogt["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_duration(html):
    soup = BeautifulSoup(html, "html.parser")
    # label
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(Duration|Thời\s*lượng|Thời\s*gian\s*bay)\s*[:\-]?\s*([^.]{1,40})", text, re.I)
    if m:
        return fmt_duration_text(m.group(2))
    # common patterns
    m = re.search(r"\b(\d{1,2}h(?:\s*\d{1,2}m)?)\b", text, re.I)
    if m: return fmt_duration_text(m.group(1))
    m = re.search(r"\b(\d{1,2}:\d{2})\b", text)  # hh:mm
    if m: return fmt_duration_text(m.group(1))
    m = re.search(r"\b(\d{1,3})\s*(minutes|min)\b", text, re.I)
    if m: return fmt_duration_text(f"{m.group(1)}m")
    return ""

def fallback_extract_stop(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True).lower()
    if re.search(r"\b(non[-\s]?stop|direct)\b", text):
        return "Nonstop"
    m = re.search(r"\b(\d+)\s*stop(s)?\b", text)
    if m:
        n = int(m.group(1))
        return "1 stop" if n == 1 else f"{n} stops"
    # transit/layover hints
    if "layover" in text or "transit" in text or "connecting" in text:
        return "1+ stops"
    return ""

def fallback_extract_price(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org price
    p = soup.select_one("[itemprop='price'], meta[itemprop='price']")
    if p:
        val = p.get("content") or p.get_text()
        cur = ""
        cnode = soup.select_one("[itemprop='priceCurrency'], meta[itemprop='priceCurrency']")
        if cnode:
            cur = (cnode.get("content") or cnode.get_text() or "").strip()
        if val:
            return f"{cur} {val}".strip()
    return parse_price(soup.get_text(" ", strip=True))

def fallback_extract_departure_time(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org
    for sel in ["[itemprop='departureTime']", "meta[itemprop='departureTime']", "time.departure", "time[datetime*='T']"]:
        node = soup.select_one(sel)
        if node:
            val = node.get("content") or node.get("datetime") or node.get_text()
            if val:
                norm = normalize_datetime(val)
                if norm: return norm
    # labels
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(Departure|Depart|Khởi hành)\s*[:\-]?\s*([^.]{1,40})", text, re.I)
    if m:
        return normalize_datetime(m.group(2))
    # HH:MM trước/sau airport code
    m = re.search(r"\b(\d{1,2}:\d{2}\s*(AM|PM)?)\s*(?:from|at|–|-)\s*[A-Z]{3}\b", text, re.I)
    if m:
        return normalize_datetime(m.group(1))
    return ""

def fallback_extract_arrival_time(html):
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["[itemprop='arrivalTime']", "meta[itemprop='arrivalTime']", "time.arrival", "time[datetime*='T']"]:
        node = soup.select_one(sel)
        if node:
            val = node.get("content") or node.get("datetime") or node.get_text()
            if val:
                norm = normalize_datetime(val)
                if norm: return norm
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(Arrival|Arrive|Đến nơi)\s*[:\-]?\s*([^.]{1,40})", text, re.I)
    if m:
        return normalize_datetime(m.group(2))
    m = re.search(r"\bto\s*[A-Z]{3}\s*(?:at|–|-)\s*(\d{1,2}:\d{2}\s*(AM|PM)?)\b", text, re.I)
    if m:
        return normalize_datetime(m.group(1))
    return ""

def fallback_extract_airline(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org Airline name
    node = soup.select_one("[itemprop='airline'] [itemprop='name'], [itemprop='airline'], meta[itemprop='airline']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()
    # label Airline
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Airline|Hãng bay)[:\s]+([^\n]{2,80})", text, re.I)
    if m: return m.group(2).strip()
    # heuristics theo title
    title = ""
    if soup.title and soup.title.get_text():
        title += " " + soup.title.get_text(strip=True)
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"):
        title += " " + ogt["content"].strip()
    for brand in AIRLINE_HINTS:
        if brand.lower() in title.lower():
            return brand
    return ""

# ==== TẠO PROMPT CHO LLAMA (domain FLIGHTS) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following flight/itinerary page text, extract these fields and return them in JSON format:\n"
        "- `name`: flight display name, e.g., 'Vietnam Airlines VN236' or 'VN236 SGN→HAN'.\n"
        "- `duration`: total flight duration (compact like '2h 15m' or '1h').\n"
        "- `stop`: 'Nonstop', '1 stop', '2 stops', etc.\n"
        "- `price`: total fare (include currency if present).\n"
        "- `departure_time`: departure date/time; prefer ISO like 'YYYY-MM-DDTHH:MM' if available, else 'HH:MM'.\n"
        "- `arrival_time`: arrival date/time; same format rule as departure_time.\n"
        "- `airline`: operating carrier name.\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: name, duration, stop, price, departure_time, arrival_time, airline (all strings), "
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
    extracted = {"name":"", "duration":"", "stop":"", "price":"", "departure_time":"", "arrival_time":"", "airline":""}
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
    writer = csv.DictWriter(out_f, fieldnames=["filename","name","duration","stop","price","departure_time","arrival_time","airline"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="✈️ Đang xử lý các HTML chuyến bay"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name":"","duration":"","stop":"","price":"","departure_time":"","arrival_time":"","airline":""})
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
            if not extracted.get("duration"):
                extracted["duration"] = fallback_extract_duration(html)
            else:
                extracted["duration"] = fmt_duration_text(extracted["duration"])
            if not extracted.get("stop"):
                extracted["stop"] = fallback_extract_stop(html)
            if not extracted.get("price"):
                extracted["price"] = fallback_extract_price(html)
            if not extracted.get("departure_time"):
                extracted["departure_time"] = fallback_extract_departure_time(html)
            else:
                extracted["departure_time"] = normalize_datetime(extracted["departure_time"])
            if not extracted.get("arrival_time"):
                extracted["arrival_time"] = fallback_extract_arrival_time(html)
            else:
                extracted["arrival_time"] = normalize_datetime(extracted["arrival_time"])
            if not extracted.get("airline"):
                extracted["airline"] = fallback_extract_airline(html)

            extracted["filename"] = filename
            writer.writerow(extracted)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "name":"","duration":"","stop":"","price":"","departure_time":"","arrival_time":"","airline":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
