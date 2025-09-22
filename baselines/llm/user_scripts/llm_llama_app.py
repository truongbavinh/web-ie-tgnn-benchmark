import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "app_html"                 # Thư mục chứa các file HTML
CSV_FILE_LIST = "file_list3.csv"       # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_app3.csv"

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

# ==== LÀM SẠCH HTML (ưu tiên trang Ứng dụng: Play Store / App Store / SoftwareApplication) ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ thẻ nhiễu
    for tag in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript"]):
        tag.decompose()
    for m in soup.select("meta[name='description'], meta[name='keywords']"):
        m.decompose()

    # Ưu tiên các block thường gặp: schema.org SoftwareApplication/MobileApplication,
    # vùng header app, thông tin nhà phát triển, đánh giá, danh mục...
    css_selectors = [
        "[itemtype*='schema.org/SoftwareApplication']",
        "[itemtype*='schema.org/MobileApplication']",
        "div.app-details", # Phổ biến cho trang chi tiết ứng dụng
        "div.app-info",
        "div.app-description",
        "section.app-overview",
        "div.main-content",
        "article.app-page",
        "div#app-main-content",
        "div.details-section", # Common on Play Store
        "div.description", # Common for app description
        "div.content", # General content wrapper
        "div[itemprop='description']", # Often used for app description
        "div.id-app-title", # Title on some app stores
        "div.score-container", # Ratings container
        "div.developer-info", # Developer section
        "div.meta-info" # General metadata like category, OS
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

# ==== CÁC HÀM FALLBACK (lấy trực tiếp từ HTML nếu LLM bỏ sót) ====
SOCIAL_DOMAINS = ("facebook.com","twitter.com","instagram.com","linkedin.com","youtube.com","tiktok.com","x.com")

def get_canonical_url(soup):
    link_canon = soup.select_one("link[rel='canonical']")
    if link_canon and link_canon.get("href"):
        return link_canon["href"].strip()
    og = soup.select_one("meta[property='og:url']")
    if og and og.get("content"):
        return og["content"].strip()
    return ""

def fallback_extract_name(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # ưu tiên meta/title/h1
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def fallback_extract_rating(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # schema.org ratingValue
    meta_rating = soup.select_one("[itemprop='ratingValue'], meta[itemprop='ratingValue']")
    if meta_rating:
        val = meta_rating.get("content") or meta_rating.get_text()
        if val: return val.strip()

    # Tìm pattern số 0-5, có thể kèm "/5", "stars", "rating"
    text = soup.get_text(" ", strip=True)
    # ví dụ: 4.6 out of 5, 4.8/5, Rating: 4.7, ★★★★☆
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", text)
    if not m:
        m = re.search(r"(?:rating[:\s]?|rated[:\s]?|\bscore[:\s]?)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1)
        try:
            x = float(candidate)
            if 0.0 <= x <= 5.0:
                return f"{x}"
        except:
            pass

    # Đếm sao unicode (rất thô): ★★★★★ (5), ★★★★☆ (4)...
    stars = re.findall(r"[★☆]", text)
    if stars:
        filled = text.count("★")
        if filled:
            return f"{min(filled,5)}"
    return ""

def fallback_extract_category(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # schema.org applicationCategory
    node = soup.select_one("[itemprop='applicationCategory'], meta[itemprop='applicationCategory']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()
    # Google Play: a dẫn tới /store/apps/category/<CAT>
    a = soup.select_one("a[href*='/store/apps/category/']")
    if a and a.get_text(strip=True):
        return a.get_text(strip=True)
    # App Store: thường có "Category" kèm giá trị bên cạnh
    label = soup.find(string=re.compile(r"Category", re.I))
    if label and label.parent:
        nxt = label.parent.find_next(text=True)
        if isinstance(nxt, str):
            return nxt.strip()
    return ""

def fallback_extract_developer(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # schema.org author
    node = soup.select_one("[itemprop='author'], meta[itemprop='author']")
    if node:
        val = node.get("content") or node.get_text()
        if val: return val.strip()
    # Google Play: "Developer" block
    dev_label = soup.find(string=re.compile(r"Developer", re.I))
    if dev_label and dev_label.parent:
        # thử lấy text kế bên
        sib_texts = dev_label.parent.get_text(" ", strip=True)
        # lấy từ sau chữ Developer:
        m = re.search(r"Developer[:\s]*(.+)", sib_texts, re.I)
        if m:
            return m.group(1).strip()
    # App Store: nhà phát triển thường là link dưới tên app
    candidates = [a.get_text(strip=True) for a in soup.select("a[href*='developer']") if a.get_text(strip=True)]
    if candidates:
        return candidates[0]
    return ""

def fallback_extract_os(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    url = get_canonical_url(soup).lower()
    text = soup.get_text(" ", strip=True).lower()
    # Dựa trên domain
    if "play.google.com" in url:
        return "Android"
    if "apps.apple.com" in url:
        return "iOS"
    # Dựa trên nội dung
    if "requires android" in text or "android" in text:
        return "Android"
    if "requires ios" in text or "ios" in text or "ipad" in text or "iphone" in text:
        return "iOS"
    if "macos" in text:
        return "macOS"
    if "windows" in text:
        return "Windows"
    return ""

# ==== TẠO PROMPT CHO LLAMA (domain APP) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # giới hạn để an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "From the following app page text, extract these fields and return them in JSON format:\n"
        "- `name`: the app's name.\n"
        "- `rating`: average user rating on a 0-5 scale (as a string).\n"
        "- `category`: app category/genre.\n"
        "- `developer`: developer or publisher name.\n"
        "- `os`: primary operating system (e.g., Android, iOS, Windows, macOS).\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: name, rating, category, developer, os (all strings), "
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
    extracted = {"name":"", "rating":"", "category":"", "developer":"", "os":""}
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
    writer = csv.DictWriter(out_f, fieldnames=["filename","name","rating","category","developer","os"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="📱 Đang xử lý các HTML ứng dụng"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name":"","rating":"","category":"","developer":"","os":""})
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
            if not extracted.get("category"):
                extracted["category"] = fallback_extract_category(html)
            if not extracted.get("developer"):
                extracted["developer"] = fallback_extract_developer(html)
            if not extracted.get("os"):
                extracted["os"] = fallback_extract_os(html)

            # Chuẩn hoá rating về chuỗi trong khoảng 0..5 (nếu bắt được)
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
            writer.writerow({"filename": filename, "name":"","rating":"","category":"","developer":"","os":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
