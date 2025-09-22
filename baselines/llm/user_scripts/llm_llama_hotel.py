import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "hotels_html"              # Thư mục chứa các file HTML khách sạn
CSV_FILE_LIST = "file_list3.csv"      # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_hotel3.csv"

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
    print("Đảm bảo đã cài 'transformers'/'torch', có GPU (hoặc đủ RAM) và quyền truy cập model trên HF.")
    raise

# ==== LÀM SẠCH HTML (ưu tiên trang khách sạn) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Ưu tiên các block thường gặp: schema.org Hotel/LodgingBusiness, sections rating/price/amenities/address
    css_selectors = [
        "[itemtype*='schema.org/Hotel']",
        "[itemtype*='schema.org/LodgingBusiness']",
        "div.uitk-layout-grid-item-has-column-start, div.Northstar, main, div#hotel",
        "div.JjjA-main, div.container-fluid ContentWrapper, div.page_detailMain__9AGj9, div#basiclayout",
        "div#content, div.hotel-details-page, .price, [itemprop='offers']",
        "section.amenities, div.amenities, ul.amenities",
        "section.location, div.location, [itemprop='address']",
        "h1[itemprop='name'], h1.hotel-name, h1.property-name",
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

# ==== TIỆN ÍCH ====
def dedupe_join(items, limit=None):
    seen = set()
    out = []
    for x in items:
        x = re.sub(r"\s+", " ", x).strip(" •,-|")
        if not x: continue
        key = x.lower()
        if key in seen: continue
        seen.add(key)
        out.append(x)
        if limit and len(out) >= limit:
            break
    return ", ".join(out)

# ==== FALLBACK (nếu LLM bỏ sót) ====
def fallback_extract_name(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("h1[itemprop='name'], h1.hotel-name, h1.property-name")
    if node: return node.get_text(strip=True)
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"): return ogt["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_location(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org PostalAddress
    addr_node = soup.select_one("[itemprop='address']")
    if addr_node:
        parts = []
        for sel in ["[itemprop='streetAddress']","[itemprop='addressLocality']",
                    "[itemprop='addressRegion']","[itemprop='postalCode']","[itemprop='addressCountry']"]:
            n = addr_node.select_one(sel)
            if n:
                v = n.get("content") or n.get_text()
                if v: parts.append(v.strip())
        if parts:
            return dedupe_join(parts)
        # nếu address chỉ là 1 khối text
        text = addr_node.get_text(" ", strip=True)
        if text: return text

    # Label "Address/Địa chỉ"
    text_all = soup.get_text("\n", strip=True)
    m = re.search(r"(Address|Địa chỉ)[:\s]+([^\n]{3,160})", text_all, re.I)
    if m: return m.group(2).strip()

    # Khối class/id phổ biến
    cand = soup.select_one(".address, #address, .location, #location")
    if cand and cand.get_text(strip=True):
        return cand.get_text(" ", strip=True)
    return ""

def fallback_extract_rating(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org ratingValue
    meta_rating = soup.select_one("[itemprop='ratingValue'], meta[itemprop='ratingValue']")
    if meta_rating:
        val = meta_rating.get("content") or meta_rating.get_text()
        if val: return _normalize_rating(val)

    text = soup.get_text(" ", strip=True)
    # 4.6/5, 4.8 out of 5
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", text)
    if not m:
        m = re.search(r"(?:rating[:\s]?|rated[:\s]?|\bscore[:\s]?)\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        return _normalize_rating(m.group(1))

    # ★★★★☆
    filled = text.count("★")
    if filled:
        return _normalize_rating(str(min(filled,5)))
    return ""

def _normalize_rating(s):
    try:
        x = float(re.findall(r"\d+(?:\.\d+)?", str(s))[0])
        x = max(0.0, min(5.0, x))
        return f"{x}"
    except:
        return str(s).strip()

def fallback_extract_price(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org price / priceRange
    p = soup.select_one("[itemprop='price'], meta[itemprop='price']")
    if p:
        val = p.get("content") or p.get_text()
        cur = ""
        cnode = soup.select_one("[itemprop='priceCurrency'], meta[itemprop='priceCurrency']")
        if cnode:
            cur = (cnode.get("content") or cnode.get_text() or "").strip()
        if val:
            return f"{cur} {val}".strip()
    pr = soup.select_one("[itemprop='priceRange'], meta[itemprop='priceRange']")
    if pr:
        val = pr.get("content") or pr.get_text()
        if val: return val.strip()

    # Regex tiền tệ + ngắn gọn "per night"
    text = soup.get_text(" ", strip=True)
    m = re.search(r"((?:USD|US\$|\$|€|£|¥|₩|₫|VND|AUD|CAD|JPY|NT\$|SGD|HKD)\s?\d[\d., ]+(?:\s*(?:/|per)\s*(?:night|đêm))?)", text, re.I)
    if m: return m.group(1).strip()
    m = re.search(r"(Price|Giá|Rate|Giá phòng)\s*[:\-]?\s*([^|,\n]{1,60})", text, re.I)
    if m: return f"{m.group(1)}: {m.group(2).strip()}"
    return ""

def fallback_extract_amenities(html):
    soup = BeautifulSoup(html, "html.parser")
    vals = []

    # schema.org amenityFeature -> name
    for n in soup.select("[itemprop='amenityFeature'] [itemprop='name'], [itemprop='amenityFeature']"):
        v = n.get("content") or n.get_text()
        if v: vals.append(v)

    # Khối .amenities, danh sách li
    for blk in soup.select(".amenities, #amenities, ul.amenities, .features, .hotel-amenities, .property-amenities"):
        for li in blk.select("li"):
            t = li.get_text(" ", strip=True)
            if t: vals.append(t)

    # Các icon/label có aria-label/title
    for el in soup.select("[aria-label*='amenit' i], [title*='amenit' i], [class*='amenit' i], [data-testid*='amenit' i]"):
        for attr in ("aria-label","title","data-testid"):
            v = el.get(attr)
            if v: vals.append(v)
        t = el.get_text(" ", strip=True)
        if t: vals.append(t)

    # Dọn trùng & rút gọn
    cleaned = []
    for v in vals:
        v = re.sub(r"\s+", " ", v)
        v = re.sub(r"(amenit(?:y|ies)|feature[s]?|tiện nghi)[:\- ]*", "", v, flags=re.I)
        v = v.strip(" •,-|")
        if v: cleaned.append(v)

    return dedupe_join(cleaned, limit=20)

# ==== TẠO PROMPT CHO LLAMA (domain HOTEL) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # giới hạn để an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following hotel property page text, extract these fields and return them in JSON format:\n"
        "- `name`: hotel/property name.\n"
        "- `location`: full postal address or concise location string.\n"
        "- `rating`: average user rating on a 0-5 scale (string).\n"
        "- `price`: nightly price/rate (include currency if present).\n"
        "- `amenities`: key amenities/features (join multiple by comma).\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: name, location, rating, price, amenities (all strings), "
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
    extracted = {"name":"", "location":"", "rating":"", "price":"", "amenities":""}
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
    writer = csv.DictWriter(out_f, fieldnames=["filename","name","location","rating","price","amenities"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🏨 Đang xử lý các HTML khách sạn"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name":"","location":"","rating":"","price":"","amenities":""})
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
            if not extracted.get("location"):
                extracted["location"] = fallback_extract_location(html)
            if not extracted.get("rating"):
                extracted["rating"] = fallback_extract_rating(html)
            else:
                extracted["rating"] = _normalize_rating(extracted["rating"])
            if not extracted.get("price"):
                extracted["price"] = fallback_extract_price(html)
            if not extracted.get("amenities"):
                extracted["amenities"] = fallback_extract_amenities(html)

            extracted["filename"] = filename
            writer.writerow(extracted)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "name":"","location":"","rating":"","price":"","amenities":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
