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
HTML_DIR = "scholarships_html"         # Thư mục chứa các file HTML học bổng
CSV_FILE_LIST = "file_list3.csv"       # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_scholarships3.csv"

# Dùng Llama 3.1 Instruct 8B
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ==== TẢI MÔ HÌNH ====
print(f"Đang tải mô hình LLM: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

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

# ==== LÀM SẠCH HTML (ưu tiên trang Scholarship) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Ưu tiên các block học bổng: schema.org/Scholarship, sections amount/deadline/award/provider
    css_selectors = [
        "[itemtype*='schema.org/Scholarship']",
        "article.scholarship-details, section.content, div.scholarship-overview, div.program-details",
        "div.cb-main-content, div.scholarship-data, div#main-content, div.content-area, div.col-sm-8"
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

# ==== TIỆN ÍCH CHUẨN HOÁ ====
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], 1
)}
MONTHS_ABBR = {m[:3].lower(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1
)}

def normalize_date(s: str) -> str:
    """Chuẩn hoá về 'YYYY-MM-DD' nếu parse được; nếu gặp 'Rolling/Varies/TBD' thì giữ nguyên."""
    if not s: return ""
    t = re.sub(r"\s+", " ", s.strip())
    if re.search(r"\b(rolling|varies|tbd|ongoing|đang diễn ra|không cố định)\b", t, re.I):
        return t

    # YYYY-MM-DD / YYYY/MM/DD
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", t)
    if m:
        y, mn, d = map(int, m.groups())
        try: return datetime(y, mn, d).strftime("%Y-%m-%d")
        except: pass

    # DD/MM/YYYY hoặc DD-MM-YYYY
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", t)
    if m:
        d, mn, y = map(int, m.groups())
        try: return datetime(y, mn, d).strftime("%Y-%m-%d")
        except: pass

    # Month DD, YYYY  hoặc DD Month YYYY
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b", t)
    if m:
        mon = MONTHS.get(m.group(1).lower()) or MONTHS_ABBR.get(m.group(1)[:3].lower())
        if mon:
            try: return datetime(int(m.group(3)), mon, int(m.group(2))).strftime("%Y-%m-%d")
            except: pass
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", t)
    if m:
        mon = MONTHS.get(m.group(2).lower()) or MONTHS_ABBR.get(m.group(2)[:3].lower())
        if mon:
            try: return datetime(int(m.group(3)), mon, int(m.group(1))).strftime("%Y-%m-%d")
            except: pass

    return t

def dedupe_join(items, limit=None):
    seen = set()
    out = []
    for x in items:
        x = re.sub(r"\s+", " ", (x or "")).strip(" •,;|-")
        if not x: continue
        k = x.lower()
        if k in seen: continue
        seen.add(k)
        out.append(x)
        if limit and len(out) >= limit: break
    return ", ".join(out)

def parse_amount_text(text: str) -> str:
    """
    Bắt các mẫu số tiền: $5,000 ; USD 10,000 ; €2.500 ; VND 50.000.000 ; ₹50,000 ; range 'up to $10,000' ; '£3,000 - £5,000/year'
    Đồng thời nhận 'Full tuition', 'Tuition waiver', 'Stipend $1,000/month'.
    """
    if not text: return ""
    t = " ".join(text.split())

    # Ưu tiên cụm "Full tuition"/"Tuition waiver"/"stipend"
    tokens = []
    if re.search(r"\b(full\s*tuition|tuition\s*waiver)\b", t, re.I):
        tokens.append(re.search(r"\b(full\s*tuition|tuition\s*waiver)\b", t, re.I).group(1).title())
    if re.search(r"\bstipend\b", t, re.I):
        # bắt cụm có tiền đi kèm stipend (nếu có)
        m = re.search(r"(stipend[^.,;:]{0,40}?(?:USD|US\$|\$|€|£|¥|₩|₫|VND|AUD|CAD|JPY|NT\$|SGD|HKD)\s?\d[\d., ]+(?:\s*/\s*(?:month|mo|year|yr))?)", t, re.I)
        tokens.append(m.group(1)) if m else tokens.append("Stipend")

    # Bắt tiền tệ + số, có thể kèm '/month' '/year'
    cur = r"(?:USD|US\$|\$|€|£|¥|₩|₫|VND|AUD|CAD|JPY|NT\$|SGD|HKD|₹|INR)"
    num = r"\d[\d., ]+"
    per = r"(?:\s*/\s*(?:month|mo|year|yr|semester|term))?"
    # range hoặc 'up to'
    patterns = [
        rf"(up to\s*{cur}\s?{num}{per})",
        rf"({cur}\s?{num}\s*[-–]\s*{cur}?\s?{num}{per})",
        rf"({cur}\s?{num}{per})",
    ]
    for p in patterns:
        for m in re.finditer(p, t, re.I):
            tokens.append(m.group(1))

    return dedupe_join(tokens, limit=3)

# ==== FALLBACK (nếu LLM bỏ sót) ====
def fallback_extract_title(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("h1[itemprop='name'], h1.scholarship-title, h1.title")
    if node: return node.get_text(strip=True)
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"): return ogt["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_provider(html):
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["[itemprop='provider'] [itemprop='name']","[itemprop='provider']",
                "[itemprop='sponsor'] [itemprop='name']","[itemprop='sponsor']",
                "[itemprop='funder'] [itemprop='name']","[itemprop='funder']",
                ".provider",".sponsor","#provider","#sponsor",".organization",".org"]:
        node = soup.select_one(sel)
        if node:
            val = node.get("content") or node.get_text()
            if val: return val.strip()
    # Label
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Provider|Sponsor|Tổ chức|Đơn vị cấp)[:\s]+([^\n]{2,120})", text, re.I)
    if m: return m.group(2).strip()
    return ""

def fallback_extract_amount(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org amount (MonetaryAmount)
    node = soup.select_one("[itemprop='amount'], meta[itemprop='amount'], [itemprop='value']")
    if node:
        val = node.get("content") or node.get_text()
        if val:
            # có thể chỉ là số → thử bắt currency xung quanh
            around = node.parent.get_text(" ", strip=True) if node.parent else val
            got = parse_amount_text(around)
            return got or val.strip()
    # Text
    return parse_amount_text(soup.get_text(" ", strip=True))

def fallback_extract_deadline(html):
    soup = BeautifulSoup(html, "html.parser")
    # time[datetime], itemprop=deadline/applicationDeadline
    for sel in ["time[datetime]","[itemprop='deadline']","meta[itemprop='deadline']",
                "[itemprop='applicationDeadline']","meta[itemprop='applicationDeadline']"]:
        node = soup.select_one(sel)
        if node:
            val = node.get("datetime") or node.get("content") or node.get_text()
            if val:
                nd = normalize_date(val)
                if nd: return nd
    # Label
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Deadline|Application deadline|Hạn chót|Hạn nộp)[:\s]+([^\n]{4,60})", text, re.I)
    if m:
        return normalize_date(m.group(2).strip())
    # 'Rolling/Varies'
    m = re.search(r"\b(Rolling|Varies|TBD|Ongoing|Đang diễn ra|Không cố định)\b", text, re.I)
    if m:
        return m.group(1)
    return ""

def fallback_extract_award(html):
    soup = BeautifulSoup(html, "html.parser")
    vals = []
    # khu award/benefits/coverage
    for blk in soup.select(".award, #award, .benefits, #benefits, .coverage, #coverage, .what-you-get, .scholarship-benefits"):
        t = blk.get_text(" ", strip=True)
        if t: vals.append(t)
        for li in blk.select("li"):
            lt = li.get_text(" ", strip=True)
            if lt: vals.append(lt)
    # label
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Award|Benefits|Quyền lợi|Học bổng bao gồm)[:\s]+(.+)", text, re.I)
    if m:
        vals.append(m.group(2))
    # Nếu rỗng, tái sử dụng amount_text + từ khoá
    if not vals:
        t = soup.get_text(" ", strip=True)
        base = parse_amount_text(t)
        extras = []
        for kw in ["tuition waiver","full tuition","accommodation","housing","monthly allowance","stipend","health insurance","travel grant","airfare","books","fees"]:
            if re.search(rf"\b{re.escape(kw)}\b", t, re.I):
                extras.append(kw.title())
        vals = [", ".join([base] + extras) if base or extras else ""]
    return dedupe_join(vals, limit=1)

# ==== TẠO PROMPT CHO LLAMA (domain SCHOLARSHIPS) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following scholarship page text, extract these fields and return them in JSON format:\n"
        "- `title`: official scholarship title.\n"
        "- `provider`: sponsoring/funding organization or university.\n"
        "- `amount`: monetary value (support currency, ranges, 'up to', or phrases like 'Full tuition'). Keep it concise.\n"
        "- `deadline`: application deadline; prefer 'YYYY-MM-DD' where possible; otherwise keep the given phrase (e.g., 'Rolling').\n"
        "- `award`: short description of what is awarded/covered (e.g., 'Full tuition, stipend $1,000/month').\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: title, provider, amount, deadline, award (all strings), "
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
    extracted = {"title":"", "provider":"", "amount":"", "deadline":"", "award":""}
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
    writer = csv.DictWriter(out_f, fieldnames=["filename","title","provider","amount","deadline","award"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🎓 Đang xử lý các HTML học bổng"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "title":"","provider":"","amount":"","deadline":"","award":""})
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
            if not extracted.get("title"):
                extracted["title"] = fallback_extract_title(html)
            if not extracted.get("provider"):
                extracted["provider"] = fallback_extract_provider(html)
            if not extracted.get("amount"):
                extracted["amount"] = fallback_extract_amount(html)
            if not extracted.get("deadline"):
                extracted["deadline"] = fallback_extract_deadline(html)
            else:
                extracted["deadline"] = normalize_date(extracted["deadline"])
            if not extracted.get("award"):
                extracted["award"] = fallback_extract_award(html)

            extracted["filename"] = filename
            writer.writerow(extracted)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "title":"","provider":"","amount":"","deadline":"","award":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
