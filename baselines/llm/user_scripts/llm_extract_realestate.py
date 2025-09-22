import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "realestate_html" # Đảm bảo thư mục này chứa các file HTML về bất động sản của bạn
CSV_FILE_LIST = "file_list3.csv" # Đảm bảo file này chứa cột "filename" với tên các file HTML
OUTPUT_CSV = "llm_output_realestate3.csv" # Tên file output

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
    Cố gắng tìm phần nội dung chính của thông tin bất động sản.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ các thẻ không chứa nội dung chính hoặc gây nhiễu
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript", "meta[name='description']"]):
        script_or_style.decompose()

    main_content = None

    # Các bộ chọn tiềm năng cho các trang bất động sản.
    # Bạn có thể cần kiểm tra HTML của các trang web cụ thể để tinh chỉnh thêm.
    selectors = [
        "div.property-detail", # Phổ biến cho trang chi tiết bất động sản
        "div.listing-info",
        "div.property-description",
        "section.property-main",
        "div.main-content",
        "article.property-page",
        "div#property-overview",
        "div.address-section", # Nơi chứa địa chỉ
        "div.price-section", # Nơi chứa giá
        "div.facts-and-features", # Nơi chứa thông tin về phòng ngủ, phòng tắm, diện tích
        "div.beds-baths-sqft", # Một bộ chọn phổ biến cho giường, tắm, diện tích
        "div.details-section", # Phần chi tiết
        "h1.property-title", # Tiêu đề/tên bất động sản
        "span.property-address", # Địa chỉ
        "div[itemprop='offers']", # Schema.org cho giá
        "div[itemprop='description']", # Schema.org cho mô tả
        "span[itemprop='numberOfBedrooms']", # Schema.org cho số phòng ngủ
        "span[itemprop='numberOfBathroomsTotal']", # Schema.org cho số phòng tắm
        "span[itemprop='floorSize']", # Schema.org cho diện tích
        "div.property-data-point", # Các điểm dữ liệu chung
        "div.re-DetailContent", # Bộ chọn từ một số trang lớn
        "div.property-attributes", # Các thuộc tính của bất động sản
        "div.c-listing-detail__main" # Một cấu trúc chung khác
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
        f"[INST] Extract the following fields from the provided real estate property information text and return them in JSON format. Ensure all values are plain strings.\n"
        f"- `title`: The title or headline of the property listing (e.g., 'Spacious Family Home with Garden', 'Modern Apartment in City Center', 'Land for Sale with Ocean View'). Return an empty string if not available.\n"
        f"- `location`: The full address or general geographical location of the property (e.g., '123 Main St, Anytown, CA', 'District 1, Ho Chi Minh City', 'Near Central Park, New York'). Return an empty string if not available.\n"
        f"- `price`: The selling or rental price of the property, including currency if specified (e.g., '$500,000', '15,000,000 VND/month', '£1,200,000'). Return an empty string if not available.\n"
        f"- `area`: The total area of the property, including units (e.g., '1,500 sq ft', '150 m²', '1000 sq m'). Return an empty string if not available.\n"
        f"- `bedrooms`: The number of bedrooms in the property (e.g., '3', 'Four bedrooms'). Return an empty string if not available.\n"
        f"- `bathrooms`: The number of bathrooms in the property (e.g., '2.5', 'Two bathrooms', '1 full, 1 half'). Return an empty string if not available.\n\n"
        f"Here is the real estate property text:\n"
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
        "title": "",
        "location": "",
        "price": "",
        "area": "",
        "bedrooms": "",
        "bathrooms": ""
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
        expected_keys = ["title", "location", "price", "area", "bedrooms", "bathrooms"]

        for key in expected_keys:
            extracted_data[key] = safe_str_strip(data.get(key))

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Lỗi phân tích JSON: {e}")
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
    writer = csv.DictWriter(out_f, fieldnames=["filename", "title", "location", "price", "area", "bedrooms", "bathrooms"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "title": "", "location": "", "price": "", "area": "", "bedrooms": "", "bathrooms": ""})
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
            writer.writerow({"filename": filename, "title": "", "location": "", "price": "", "area": "", "bedrooms": "", "bathrooms": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")