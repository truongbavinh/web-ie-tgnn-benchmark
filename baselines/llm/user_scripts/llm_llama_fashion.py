import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "fashion_html"             # Thư mục chứa các file HTML thời trang
CSV_FILE_LIST = "file_list3.csv"       # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_fashion3.csv"

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

# ==== LÀM SẠCH HTML (ưu tiên trang thời trang: product/specs/variants) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Các block thường gặp: schema.org Product/Offer/Brand; info/price/variants/specs
    css_selectors = [
        "[itemtype*='schema.org/Product']",
        "div.wrapper", # Phổ biến cho trang chi tiết sản phẩm
        "main#content",
        "div.item-description",
        "main.de352b",
        "main#main",
        "div#product-main-info",
        "div.main-content",
        "article.product-page",
        "section.product-section",
        "div.col-md-6.product-detail-info", # Ví dụ từ cấu trúc chia cột
        "div.wrapper", # Bộ chọn chung, đôi khi hữu ích
        "div.container-fluid.ContentWrapper", # Bộ chọn chung khác
        "div.zds-modal.zds-dialog", # Nếu thông tin nằm trong modal/popup
        "div[data-component='product-details']", # Các trang dùng framework
        "div.css-xxxxxx", # Các lớp CSS tự động sinh ra có thể cần phân tích
        "div.product-attributes", # Nơi chứa vật liệu, màu sắc, kích thước
        "div.price-info" # Nơi chứa giá
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
def dedupe_join(items):
    seen = set()
    out = []
    for x in items:
        x = re.sub(r"\s+", " ", x).strip()
        if not x: continue
        if x.lower() in seen: continue
        seen.add(x.lower())
        out.append(x)
    return ", ".join(out)

# ==== FALLBACK (nếu LLM bỏ sót) ====
def fallback_extract_name(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("h1[itemprop='name'], h1.product-title, h1.product-name")
    if node: return node.get_text(strip=True)
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"): return ogt["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_price(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org price(+currency)
    p = soup.select_one("[itemprop='price'], meta[itemprop='price']")
    if p:
        val = p.get("content") or p.get_text()
        cur = ""
        cnode = soup.select_one("[itemprop='priceCurrency'], meta[itemprop='priceCurrency']")
        if cnode:
            cur = (cnode.get("content") or cnode.get_text() or "").strip()
        if val:
            return f"{cur} {val}".strip()
    # low/high price
    lp = soup.select_one("[itemprop='lowPrice']")
    hp = soup.select_one("[itemprop='highPrice']")
    if lp or hp:
        lpv = (lp.get("content") if lp and lp.get("content") else (lp.get_text().strip() if lp else ""))
        hpv = (hp.get("content") if hp and hp.get("content") else (hp.get_text().strip() if hp else ""))
        if lpv and hpv: return f"{lpv} - {hpv}"
        if lpv: return lpv
        if hpv: return hpv
    # Regex tiền tệ (USD/VND/€/£/¥/…)
    text = soup.get_text(" ", strip=True)
    m = re.search(r"((?:USD|US\$|\$|€|£|¥|₩|₫|VND|AUD|CAD|JPY|NT\$|SGD|HKD)\s?\d[\d., ]+)", text, re.I)
    if m: return m.group(1).strip()
    # Label
    m = re.search(r"(Price|Giá|Sale price|Giá khuyến mãi)\s*[:\-]?\s*([^|,\n]{1,60})", text, re.I)
    if m: return f"{m.group(1)}: {m.group(2).strip()}"
    return ""

def fallback_extract_material(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("[itemprop='material'], meta[itemprop='material']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return re.sub(r"\s+", " ", val).strip()
    # Từ khoá: Material(s)/Fabric/Composition/Chất liệu/Thành phần
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Material(?:s)?|Fabric|Composition|Chất\s*liệu|Thành\s*phần)[:\s]+([^\n]{2,160})", text, re.I)
    if m: return m.group(2).strip()
    # Danh sách gạch đầu dòng
    bullets = [li.get_text(" ", strip=True) for li in soup.select("li") if re.search(r"(cotton|poly|linen|silk|wool|da|canvas|spandex|viscose|rayon)", li.get_text(" ", strip=True), re.I)]
    if bullets:
        return dedupe_join(bullets[:3])
    return ""

def fallback_extract_color(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org color
    vals = []
    for node in soup.select("[itemprop='color'], meta[itemprop='color']"):
        val = node.get("content") or node.get_text()
        if val: vals.append(val)
    # Label Color/Màu
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Color|Màu sắc|Màu)[:\s]+([^\n]{1,120})", text, re.I)
    if m: vals.append(m.group(2))
    # Swatches/variants
    for el in soup.select("[class*='color'], [class*='swatch'], [data-color], [aria-label*='color']"):
        # ưu tiên attr
        for attr in ("data-color","aria-label","title"):
            v = el.get(attr)
            if v: vals.append(v)
        t = el.get_text(" ", strip=True)
        if t: vals.append(t)
    # Lọc gọn
    # loại mã hex, giữ tên màu nếu có
    cleaned = []
    for v in vals:
        v = re.sub(r"#?[0-9A-Fa-f]{3,6}", "", v)  # bỏ hex
        v = re.sub(r"\b(color|màu|swatch|select|option)\b", "", v, flags=re.I)
        v = re.sub(r"\s+", " ", v).strip(" :,-")
        if v: cleaned.append(v)
    return dedupe_join(cleaned)

def fallback_extract_size(html):
    soup = BeautifulSoup(html, "html.parser")
    vals = []
    # schema-ish (hiếm)
    for node in soup.select("[itemprop='size'], meta[itemprop='size']"):
        v = node.get("content") or node.get_text()
        if v: vals.append(v)
    # Select/options liên quan size
    for sel in soup.select("select[name*='size' i] option, select[id*='size' i] option"):
        t = sel.get_text(" ", strip=True)
        if t and not re.search(r"(select|choose|chọn)", t, re.I):
            vals.append(t)
    # Buttons/labels có class size/variant
    for el in soup.select("[class*='size'], .variant, .option"):
        t = el.get_text(" ", strip=True)
        if t:
            # tách theo dấu phẩy / khoảng trắng để lấy S, M, L...
            parts = [p.strip() for p in re.split(r"[,\s/|]+", t) if p.strip()]
            # lọc token hợp lệ
            for p in parts:
                if re.fullmatch(r"(XXS|XS|S|M|L|XL|XXL|XXXL|\d{2}(\.\d)?|\d{1,2}[A-Z]?|EU\d{2}|US\d{1,2}|UK\d{1,2})", p, re.I):
                    vals.append(p.upper())
    # Label 'Size: ...'
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Size|Kích cỡ|Cỡ)[:\s]+([^\n]{1,120})", text, re.I)
    if m:
        vals.append(m.group(2))
    # Dọn sạch
    cleaned = []
    for v in vals:
        v = re.sub(r"(size|kích cỡ|cỡ|select|choose|chọn)[:\s-]*", "", v, flags=re.I)
        v = re.sub(r"\s+", " ", v).strip(" ,|-")
        if v: cleaned.append(v)
    return dedupe_join(cleaned)

# ==== TẠO PROMPT CHO LLAMA (domain FASHION) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # giới hạn để an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following fashion product page text, extract these fields and return them in JSON format:\n"
        "- `name`: product name.\n"
        "- `price`: listed price (include currency if present).\n"
        "- `material`: main materials or composition (e.g., 100% Cotton; Wool, Polyester).\n"
        "- `color`: color(s); if multiple options, join with comma.\n"
        "- `size`: size options (e.g., S, M, L, 38, EU42); join with comma.\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: name, price, material, color, size (all strings), "
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
    extracted = {"name":"", "price":"", "material":"", "color":"", "size":""}
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
    writer = csv.DictWriter(out_f, fieldnames=["filename","name","price","material","color","size"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="👗 Đang xử lý các HTML thời trang"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name":"","price":"","material":"","color":"","size":""})
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
            if not extracted.get("price"):
                extracted["price"] = fallback_extract_price(html)
            if not extracted.get("material"):
                extracted["material"] = fallback_extract_material(html)
            if not extracted.get("color"):
                extracted["color"] = fallback_extract_color(html)
            if not extracted.get("size"):
                extracted["size"] = fallback_extract_size(html)

            extracted["filename"] = filename
            writer.writerow(extracted)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "name":"","price":"","material":"","color":"","size":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
