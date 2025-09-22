import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "app_html" # Đảm bảo thư mục này chứa các file HTML về ứng dụng của bạn
CSV_FILE_LIST = "file_list3.csv" # Đảm bảo file này chứa cột "filename" với tên các file HTML
OUTPUT_CSV = "llm_output_app3.csv" # Đổi tên file output để tránh ghi đè

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "auto" # device_map="auto" sẽ tự động quản lý

# ==== TẢI MÔ HÌNH ====
print(f"Đang tải mô hình LLM: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=False, temperature=0.1)
    print("Mô hình LLM đã tải xong.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("Đảm bảo bạn đã cài đặt thư viện 'transformers' và 'torch' với CUDA (nếu dùng GPU) hoặc có đủ RAM.")
    exit() # Thoát nếu không tải được mô hình

# ==== HÀM LÀM SẠCH HTML VÀ CHUYỂN ĐỔI SANG VĂN BẢN THÔ ====
def get_clean_text_from_html(html_content):
    """
    Làm sạch HTML và trích xuất văn bản thô, loại bỏ các phần tử không liên quan.
    Cố gắng tìm phần nội dung chính của thông tin ứng dụng.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ các thẻ không chứa nội dung chính hoặc gây nhiễu
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript", "meta[name='description']"]):
        script_or_style.decompose()

    main_content = None

    # Các bộ chọn tiềm năng cho các trang ứng dụng (Google Play, Apple App Store, v.v.).
    # Bạn có thể cần kiểm tra HTML của các trang web cụ thể để tinh chỉnh thêm.
    selectors = [
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

    for selector in selectors:
        # Kiểm tra theo class, id hoặc tag name
        if '.' in selector: # Là class selector
            main_content = soup.find(class_=selector.split('.')[-1])
        elif '#' in selector: # Là id selector
            main_content = soup.find(id=selector.split('#')[-1])
        else: # Là tag name selector
            main_content = soup.find(selector)

        if main_content:
            break # Tìm thấy nội dung chính, thoát vòng lặp

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        # Fallback: lấy toàn bộ văn bản từ body nếu không tìm thấy nội dung chính cụ thể
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

    # Loại bỏ khoảng trắng thừa và dòng trống lặp lại
    text = "\n".join(filter(lambda x: x.strip(), text.split("\n")))
    return text

# ==== HÀM TẠO PROMPT CHO LLM ====
def create_llm_prompt(clean_html_text):
    """
    Tạo prompt cho LLM, nhúng văn bản thô đã làm sạch.
    """
    # Giới hạn văn bản để không vượt quá context window của LLM
    short_text_for_llm = clean_html_text[:4000] # Giữ 4000 ký tự để an toàn

    prompt = (
        f"[INST] Extract the following fields from the provided app information text and return them in JSON format. Ensure all values are plain strings.\n"
        f"- `name`: The name of the application (e.g., 'Google Maps', 'TikTok', 'Zoom'). Return an empty string if not available.\n"
        f"- `rating`: The numerical rating of the app, typically out of 5 (e.g., '4.7', '3.9 stars'). Return an empty string if not available.\n"
        f"- `category`: The category or genre the app belongs to (e.g., 'Social', 'Navigation', 'Productivity', 'Games'). If multiple, provide as a comma-separated string. Return an empty string if not available.\n"
        f"- `developer`: The name of the developer or company that created the app (e.g., 'Google LLC', 'ByteDance', 'Zoom Video Communications'). Return an empty string if not available.\n"
        f"- `os`: The operating system(s) the app is available on (e.g., 'Android', 'iOS', 'Windows', 'macOS', 'Android, iOS'). If multiple, provide as a comma-separated string. Return an empty string if not available.\n\n"
        f"Here is the app text:\n"
        f"```\n{short_text_for_llm}\n```\n\n"
        f"Return the result as a JSON object, ensuring all specified keys are present, even if their values are empty strings. Ensure the JSON is valid and can be parsed. Do not include any additional text or explanations outside the JSON object.\n"
        f"[/INST]"
    )
    return prompt

# ==== HÀM XỬ LÝ ĐẦU RA JSON TỪ LLM ====
def parse_llm_response_to_json(llm_response):
    """
    Phân tích cú pháp phản hồi của LLM để trích xuất JSON.
    Xử lý các trường bị thiếu và đảm bảo định dạng đúng,
    kể cả khi LLM trả về danh sách thay vì chuỗi.
    """
    # Khởi tạo dictionary với tất cả các khóa và giá trị mặc định là chuỗi rỗng
    extracted_data = {
        "name": "",
        "rating": "",
        "category": "",
        "developer": "",
        "os": ""
    }

    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = llm_response[json_start : json_end + 1]
            else:
                raise ValueError("No valid JSON structure found in LLM response.")

        data = json.loads(json_str)

        # Helper function để xử lý an toàn giá trị: chuyển đổi sang chuỗi và strip
        def safe_str_strip(value):
            if isinstance(value, list):
                # Nếu là list, nối các phần tử thành một chuỗi cách nhau bởi ", "
                return ", ".join(map(str, value)).strip()
            elif value is None:
                # Trả về chuỗi rỗng nếu giá trị là None
                return ""
            else:
                # Chuyển đổi sang chuỗi và strip cho các kiểu dữ liệu khác
                return str(value).strip()

        # Danh sách các khóa mong muốn để đảm bảo xử lý nhất quán
        expected_keys = ["name", "rating", "category", "developer", "os"]

        for key in expected_keys:
            extracted_data[key] = safe_str_strip(data.get(key))

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ JSON Parse Error: {e}")
        print(f"LLM Response (full): {llm_response}")
        # extracted_data vẫn giữ các giá trị mặc định nếu có lỗi parsing
    return extracted_data

# ==== ĐỌC DANH SÁCH FILE HTML ====
try:
    with open(CSV_FILE_LIST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filenames = [row["filename"] for row in reader]
except FileNotFoundError:
    print(f"Lỗi: Tệp '{CSV_FILE_LIST}' không tìm thấy. Vui lòng đảm bảo tệp nằm cùng thư mục với script.")
    exit()

# ==== GHI KẾT QUẢ RA CSV ====
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["filename", "name", "rating", "category", "developer", "os"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name": "", "rating": "", "category": "", "developer": "", "os": ""})
            continue

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            clean_text = get_clean_text_from_html(html_content)
            prompt = create_llm_prompt(clean_text)
            llm_raw_response = pipe(prompt)[0]["generated_text"]
            extracted_data = parse_llm_response_to_json(llm_raw_response)

            extracted_data["filename"] = filename
            writer.writerow(extracted_data)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            writer.writerow({"filename": filename, "name": "", "rating": "", "category": "", "developer": "", "os": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")