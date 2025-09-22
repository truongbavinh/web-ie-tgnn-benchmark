import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "cooking_html"             # Thư mục chứa các file HTML recipe
CSV_FILE_LIST = "file_list3.csv"       # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_cooking3.csv"

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
    print("Đảm bảo đã cài 'transformers'/'torch', có GPU (hoặc đủ RAM) và đã được cấp quyền truy cập Llama trên HF.")
    raise

# ==== LÀM SẠCH HTML (ưu tiên trang Recipe) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Ưu tiên các block thường gặp của recipe: schema.org Recipe, rating/author/time/category...
    css_selectors = [
        "[itemtype*='schema.org/Recipe']",
        "main#main, section.o-Recipe, main.main, div.pagegrid_container__ExzOM",
        "section.recipe-main, div.content-header, section#main-content",
        "section.ingredients, section.instructions, section.method, section.overview",
        "div.rating, section.rating, div.author, section.author, div.meta, section.meta",
        "h1.recipe-title, h1[itemprop='name']",
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

# ==== CHUẨN HOÁ THỜI GIAN ====
def minutes_from_iso8601(duration: str) -> int:
    """
    Parse ISO 8601 duration (e.g., PT1H30M, P0DT45M) -> minutes (int). Return -1 nếu không parse được.
    """
    if not duration: return -1
    t = duration.upper().strip()
    # P[n]DT[n]H[n]M[n]S hoặc PT...
    m = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", t)
    if not m:
        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", t)
    if not m:
        return -1
    days = int(m.group(1) or 0) if len(m.groups()) >= 1 else 0
    hours = int(m.group(2) or 0) if len(m.groups()) >= 2 else 0
    mins  = int(m.group(3) or 0) if len(m.groups()) >= 3 else 0
    secs  = int(m.group(4) or 0) if len(m.groups()) >= 4 else 0
    total = days*24*60 + hours*60 + mins + (secs // 60)
    return total if total > 0 else -1

def minutes_from_text(s: str) -> int:
    """
    Parse thời gian text kiểu '1 hr 30 mins', '1h30', '45 minutes', '1 giờ 30 phút', v.v. -> minutes.
    """
    if not s: return -1
    t = s.lower()

    # hh:mm (e.g., 1:30)
    m = re.search(r"\b(\d{1,2}):(\d{1,2})\b", t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    # hours + minutes (English/Vietnamese)
    h = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours|giờ|gio)", t)
    mn = re.search(r"(\d+)\s*(m|min|mins|minute|minutes|phút|phut|p)", t)
    if h and mn:
        return int(h.group(1)) * 60 + int(mn.group(1))
    if h:
        return int(h.group(1)) * 60
    if mn:
        return int(mn.group(1))

    # 1h30, 2h05, 1g30
    m = re.search(r"(\d+)\s*[hHgGpP]\s*(\d{1,2})", t)
    if m:
        return int(m.group(1))*60 + int(m.group(2))

    # 'ready in 45 mins' / 'total time 20 minutes'
    m = re.search(r"(?:ready\s*in|total\s*time|tổng\s*thời\s*gian|thời\s*gian)\s*[:\-]?\s*(\d+)\s*(m|min|minutes?)", t)
    if m:
        return int(m.group(1))

    return -1

def fmt_minutes(m: int) -> str:
    if m <= 0: return ""
    h, mm = divmod(m, 60)
    return (f"{h}h {mm}m" if h else f"{mm}m").strip()

# ==== FALLBACK (nếu LLM bỏ sót) ====
def fallback_extract_name(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("h1[itemprop='name'], h1.recipe-title")
    if node: return node.get_text(strip=True)
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"): return ogt["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_rating(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org ratingValue
    meta_rating = soup.select_one("[itemprop='ratingValue'], meta[itemprop='ratingValue']")
    if meta_rating:
        val = meta_rating.get("content") or meta_rating.get_text()
        if val: return val.strip()

    text = soup.get_text(" ", strip=True)
    # 4.6/5, 4.8 out of 5
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", text)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:out\s*of\s*)?5", text, re.I)
    if m:
        try:
            x = float(m.group(1))
            if 0.0 <= x <= 5.0:
                return f"{x}"
        except:
            pass

    # ★★★★☆
    filled = text.count("★")
    if filled:
        return f"{min(filled,5)}"
    return ""

def fallback_extract_author(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org author (Person.name)
    node = soup.select_one("[itemprop='author'] [itemprop='name'], [itemprop='author'], meta[name='author']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()

    # "By <name>" / "Tác giả: <name>"
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(?:By|Tác giả)[:\s]+([^\n]{3,120})", text, re.I)
    if m: return m.group(1).strip()

    # class/id author
    cand = soup.select_one(".author, #author")
    if cand and cand.get_text(strip=True):
        return cand.get_text(strip=True)
    return ""

def fallback_extract_time(html):
    soup = BeautifulSoup(html, "html.parser")

    # schema.org totalTime (ISO 8601)
    node = soup.select_one("[itemprop='totalTime'], meta[itemprop='totalTime']")
    if node:
        val = node.get("content") or node.get_text()
        mins = minutes_from_iso8601(val)
        if mins > 0: return fmt_minutes(mins)

    # Nếu không có totalTime: lấy prepTime + cookTime
    def get_minutes_for(prop):
        n = soup.select_one(f"[itemprop='{prop}'], meta[itemprop='{prop}']")
        if n:
            v = n.get("content") or n.get_text()
            return max(minutes_from_iso8601(v), minutes_from_text(v))
        return -1

    prep = get_minutes_for("prepTime")
    cook = get_minutes_for("cookTime")
    if prep > 0 or cook > 0:
        total = (prep if prep > 0 else 0) + (cook if cook > 0 else 0)
        if total > 0: return fmt_minutes(total)

    # Tìm text "Total time: ..." / "Ready in ..."
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(?:Total\s*time|Ready\s*in|Tổng\s*thời\s*gian|Thời\s*gian)\s*[:\-]?\s*([^.]{1,40})", text, re.I)
    if m:
        mins = max(minutes_from_text(m.group(1)), minutes_from_iso8601(m.group(1)))
        if mins > 0: return fmt_minutes(mins)

    # Bắt số phút chung
    m = re.search(r"(\d+)\s*(minutes|min|phút|p)\b", text, re.I)
    if m:
        return fmt_minutes(int(m.group(1)))

    return ""

def fallback_extract_type(html):
    soup = BeautifulSoup(html, "html.parser")
    # schema.org recipeCategory (type/course), hoặc recipeCuisine (cuisine)
    node = soup.select_one("[itemprop='recipeCategory'], meta[itemprop='recipeCategory']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()
    node = soup.select_one("[itemprop='recipeCuisine'], meta[itemprop='recipeCuisine']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()

    # "Course: Dessert/Main Course/..." hoặc "Loại: ..."
    text = soup.get_text("\n", strip=True)
    m = re.search(r"(Course|Type|Loại|Danh mục)[:\s]+([^\n]{2,60})", text, re.I)
    if m: return m.group(2).strip()

    # Heuristic theo tiêu đề (ít ưu tiên)
    title = (soup.title.get_text(strip=True) if soup.title else "")
    for t in ["Dessert","Main Course","Side Dish","Appetizer","Breakfast","Beverage","Soup","Salad","Sauce"]:
        if t.lower() in title.lower():
            return t
    return ""

# ==== TẠO PROMPT CHO LLAMA (domain COOKING/RECIPE) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # giới hạn để an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following recipe page text, extract these fields and return them in JSON format:\n"
        "- `name`: recipe name.\n"
        "- `rating`: average user rating on a 0-5 scale (as a string).\n"
        "- `author`: recipe author.\n"
        "- `time`: total time to make the recipe (prefer total time; if only prep/cook are available, sum them). Use a compact format like '1h 30m' or '45m'.\n"
        "- `type`: recipe category or course (e.g., Dessert, Main Course) or cuisine if category not available.\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: name, rating, author, time, type (all strings), "
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
    extracted = {"name":"", "rating":"","author":"","time":"","type":""}
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
    writer = csv.DictWriter(out_f, fieldnames=["filename","name","rating","author","time","type"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🍳 Đang xử lý các HTML công thức nấu ăn"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name":"","rating":"","author":"","time":"","type":""})
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
            if not extracted.get("rating"):
                extracted["rating"] = fallback_extract_rating(html)
            if not extracted.get("author"):
                extracted["author"] = fallback_extract_author(html)
            if not extracted.get("time"):
                extracted["time"] = fallback_extract_time(html)
            if not extracted.get("type"):
                extracted["type"] = fallback_extract_type(html)

            # Chuẩn hoá rating về chuỗi 0..5 nếu parse được
            try:
                if extracted["rating"]:
                    x = float(re.findall(r"\d+(?:\.\d+)?", extracted["rating"])[0])
                    x = max(0.0, min(5.0, x))
                    extracted["rating"] = f"{x}"
            except Exception:
                pass

            extracted["filename"] = filename
            writer.writerow(extracted)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "name":"","rating":"","author":"","time":"","type":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
