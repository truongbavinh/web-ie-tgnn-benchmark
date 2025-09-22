import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "tourist_html" # Đảm bảo thư mục này chứa các file HTML về du lịch của bạn
CSV_FILE_LIST = "file_list3.csv" # Đảm bảo file này chứa cột "filename" với tên các file HTML
OUTPUT_CSV = "llm_output_tourist3.csv" # Đổi tên file output để tránh ghi đè

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
    Cố gắng tìm phần nội dung chính của thông tin du lịch (điểm đến, tour, khách sạn...).
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ các thẻ không chứa nội dung chính hoặc gây nhiễu
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript", "meta[name='description']"]):
        script_or_style.decompose()

    main_content = None

    # Các bộ chọn tiềm năng cho các trang du lịch (tour, điểm đến, khách sạn...).
    # Bạn có thể cần kiểm tra và tinh chỉnh thêm dựa trên cấu trúc HTML của các trang web cụ thể.
    selectors = [
        "div.tour-detail", # Phổ biến cho trang chi tiết tour
        "div.attraction-info", # Thông tin điểm tham quan
        "div.trip-description", # Mô tả chuyến đi
        "section.main-content",
        "article.page-content",
        "div.product-details", # Chung cho các sản phẩm du lịch
        "div.location-info", # Thông tin địa điểm
        "div.review-section", # Phần đánh giá
        "div.price-section", # Phần giá
        "div.duration-info", # Thông tin thời lượng
        "div#main", # Một số trang dùng ID main cho nội dung chính
        "div.container", # Bộ chọn chung
        "div[data-component='tour-details']", # Các trang dùng framework
        "div.hero-section", # Phần đầu trang thường chứa tên, ảnh, rating
        "div.overview-section" # Tổng quan
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
        f"[INST] Extract the following fields from the provided tourist attraction or tour information text and return them in JSON format. Ensure all values are plain strings.\n"
        f"- `name`: The name of the tourist attraction or tour (e.g., 'Eiffel Tower', 'Ha Long Bay Cruise', 'Grand Canyon Helicopter Tour'). Return an empty string if not available.\n"
        f"- `location`: The geographical location of the attraction or where the tour takes place (e.g., 'Paris, France', 'Quang Ninh Province, Vietnam', 'Arizona, USA'). Return an empty string if not available.\n"
        f"- `rating`: The numerical rating (e.g., '4.8/5', '5 stars', 'Excellent'). Return an empty string if not available.\n"
        f"- `price`: The price of the tour or admission fee, including currency (e.g., '$50', '2,000,000 VND per person', 'Free'). Return an empty string if not available.\n"
        f"- `duration`: The typical duration of the tour or visit (e.g., '3 hours', '2 days 1 night', 'Full day', 'Approximately 90 minutes'). Return an empty string if not available.\n\n"
        f"Here is the tourist information text:\n"
        f"```\n{short_text_for_llm}\n```\n\n"
        f"Return the result as a JSON object, ensuring all specified keys are present, even if their values are empty strings. Ensure the JSON is valid and can be parsed. Do not include any additional text or explanations outside the JSON object.\n"
        f"[/INST]"
    )
    return prompt

# ==== HÀM XỬ LÝ ĐẦU RA JSON TỪ LLM ====
def parse_llm_response_to_json(llm_response):
    """
    Phân tích cú pháp phản hồi của LLM để trích xuất JSON.
    Xử lý các trường bị thiếu và đảm bảo định dạng đúng.
    """
    # Khởi tạo dictionary với tất cả các khóa và giá trị mặc định là chuỗi rỗng
    extracted_data = {
        "name": "",
        "location": "",
        "rating": "",
        "price": "",
        "duration": ""
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
        expected_keys = ["name", "location", "rating", "price", "duration"]

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
    writer = csv.DictWriter(out_f, fieldnames=["filename", "name", "location", "rating", "price", "duration"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name": "", "location": "", "rating": "", "price": "", "duration": ""})
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
            writer.writerow({"filename": filename, "name": "", "location": "", "rating": "", "price": "", "duration": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")