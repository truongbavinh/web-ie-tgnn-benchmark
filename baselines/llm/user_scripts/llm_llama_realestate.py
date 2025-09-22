import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "realestate_html"           # Thư mục chứa các file HTML
CSV_FILE_LIST = "file_list3.csv"       # CSV có cột "filename"
OUTPUT_CSV = "llm_llama_realestate3.csv"

# Dùng Llama 3.1 Instruct 8B (hợp lý cho GPU 40GB)
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

# ==== LÀM SẠCH HTML ====
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script","style","header","footer","nav","aside","form","ins","button","iframe","noscript",
                                 "meta[name='description']"]):
        script_or_style.decompose()

    main_content = None
    selectors = [
        "div.property-detail","div.listing-info","div.details","div.Containerstyles__StyledContainer-rui__q3yf4x-0.flIOie",
        "div.main-content","div.WJG_W7faYk84nW-6sCBVi","div#property-overview","div.layout-container-desktop",
        "div.price-section","div.facts-and-features","div.beds-baths-sqft","div.details-section",
        "h1.property-title","span.property-address","div[itemprop='offers']","div[itemprop='description']",
        "span[itemprop='numberOfBedrooms']","span[itemprop='numberOfBathroomsTotal']","span[itemprop='floorSize']",
        "div.property-data-point","div.re-DetailContent","div.property-attributes","div.c-listing-detail__main"
    ]
    for selector in selectors:
        if '.' in selector:
            main_content = soup.find(class_=selector.split('.')[-1])
        elif '#' in selector:
            main_content = soup.find(id=selector.split('#')[-1])
        else:
            main_content = soup.find(selector)
        if main_content:
            break

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body")
        text = (body or soup).get_text(separator="\n", strip=True)

    text = "\n".join(filter(lambda x: x.strip(), text.split("\n")))
    return text

# ==== TẠO PROMPT CHO LLAMA (dùng chat template) ====
def create_llm_prompt_for_llama(clean_html_text, tokenizer):
    short_text = clean_html_text[:4000]  # giới hạn để an toàn context

    system_msg = (
        "You are a structured information extractor. "
        "Extract the requested fields and output ONLY a valid JSON object. "
        "All specified keys must be present; if missing, use empty strings. "
        "Wrap the JSON in triple backticks with a 'json' language tag."
    )
    user_msg = (
        "Extract the following fields from the provided real estate property information text and return them in JSON format.\n"
        "- `title`: listing title/headline.\n"
        "- `location`: full address or general location.\n"
        "- `price`: selling or rental price (include currency if present).\n"
        "- `area`: total area with units.\n"
        "- `bedrooms`: number of bedrooms.\n"
        "- `bathrooms`: number of bathrooms.\n\n"
        "Here is the text:\n"
        f"```\n{short_text}\n```\n\n"
        "Return ONLY a JSON object with keys: title, location, price, area, bedrooms, bathrooms, all as strings, "
        "and wrap it like:\n```json\n{...}\n```\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    # Biến messages thành prompt string theo đúng chat template của Llama
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_str

# ==== PARSE JSON TỪ ĐẦU RA ====
def parse_llm_response_to_json(llm_response):
    extracted = {"title":"", "location":"", "price":"", "area":"", "bedrooms":"", "bathrooms":""}
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
    writer = csv.DictWriter(out_f, fieldnames=["filename","title","location","price","area","bedrooms","bathrooms"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "title":"","location":"","price":"","area":"","bedrooms":"","bathrooms":""})
            continue

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            clean_text = get_clean_text_from_html(html)
            prompt = create_llm_prompt_for_llama(clean_text, tokenizer)

            # Quan trọng: trả về CHỈ phần sinh ra (không kèm prompt) để dễ parse
            llm_raw_response = pipe(prompt, return_full_text=False)[0]["generated_text"]
            extracted = parse_llm_response_to_json(llm_raw_response)

            extracted["filename"] = filename
            writer.writerow(extracted)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "title":"","location":"","price":"","area":"","bedrooms":"","bathrooms":""})

print(f"\n✅ Hoàn tất. Kết quả lưu tại '{OUTPUT_CSV}'")
