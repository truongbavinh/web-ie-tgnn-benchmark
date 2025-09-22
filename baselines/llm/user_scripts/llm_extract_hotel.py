import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "hotels_html" # Đã được cập nhật từ "events_html"
CSV_FILE_LIST = "file_list3.csv"
OUTPUT_CSV = "llm_output_hotel3.csv"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "auto"

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
    exit()

# ==== HÀM LÀM SẠCH HTML VÀ CHUYỂN ĐỔI SANG VĂN BẢN THÔ ====
def get_clean_text_from_html(html_content):
    """
    Làm sạch HTML và trích xuất văn bản thô, loại bỏ các phần tử không liên quan.
    Cố gắng tìm phần nội dung chính của thông tin khách sạn.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript"]):
        script_or_style.decompose()

    # Cố gắng tìm phần nội dung chính của trang khách sạn/chuyến bay.
    # Các bộ chọn này cần được điều chỉnh/bổ sung dựa trên phân tích các domain của bạn (kayak, booking, trip, agoda).
    # Đã thêm các bộ chọn chung hơn hoặc phổ biến trên các trang đặt phòng.
    main_content = None

    # Ví dụ các bộ chọn có thể xuất hiện trên các trang đặt phòng:
    main_content = soup.find("div", class_="uitk-layout-grid-item-has-column-start") \
                   or soup.find("main") \
                   or soup.find("div", class_="Northstar") \
                   or soup.find("div", class_="JjjA-main") \
                   or soup.find("div", class_="container-fluid ContentWrapper") \
                   or soup.find("div", class_="page_detailMain__9AGj9") \
                   or soup.find("div", id="basiclayout") \
                   or soup.find("div", id="content") \
                   or soup.find("div", class_="hotel-details-page") # Một class chung chung khác

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

    text = "\n".join(filter(lambda x: x.strip(), text.split("\n")))
    return text

# ==== HÀM TẠO PROMPT CHO LLM ====
def create_llm_prompt(clean_html_text):
    """
    Tạo prompt cho LLM, nhúng văn bản thô đã làm sạch.
    """
    short_text_for_llm = clean_html_text[:4000]

    prompt = (
        f"[INST] Extract the following fields from the provided hotel/product information text and return them in JSON format. Ensure all values are plain strings.\n"
        f"- `name`: The name of the hotel or product.\n"
        f"- `location`: The geographical location or address of the hotel. Return an empty string if not available.\n"
        f"- `rating`: The numerical rating of the hotel (e.g., '4.5', '5 stars'). Return an empty string if not available.\n"
        f"- `price`: The price of the hotel per night or the product price, including currency if specified (e.g., '$150', '2,000,000 VND'). Return an empty string if not available.\n"
        f"- `amenities`: A comma-separated list of amenities available (e.g., 'Wifi, Pool, Parking, Breakfast'). If amenities are listed as a paragraph, summarize them. Return an empty string if not available.\n\n"
        f"Here is the text:\n"
        f"```\n{short_text_for_llm}\n```\n\n"
        f"Return the result as a JSON object, ensuring all specified keys are present, even if their values are empty strings. Do not include any additional text or explanations outside the JSON object.\n"
        f"[/INST]"
    )
    return prompt

# ==== HÀM XỬ LÝ ĐẦU RA JSON TỪ LLM ====
def parse_llm_response_to_json(llm_response):
    """
    Phân tích cú pháp phản hồi của LLM để trích xuất JSON.
    Xử lý các trường bị thiếu và đảm bảo định dạng đúng.
    """
    # Khởi tạo dictionary với các giá trị mặc định là chuỗi rỗng
    extracted_data = {"name": "", "location": "", "rating": "", "price": "", "amenities": ""}

    try:
        # Tìm phần JSON trong phản hồi của LLM
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback nếu không có ```json, tìm dấu { đầu tiên và } cuối cùng
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = llm_response[json_start : json_end + 1]
            else:
                raise ValueError("No valid JSON structure found in LLM response.")

        data = json.loads(json_str)

        # Danh sách các khóa mong muốn để đảm bảo xử lý nhất quán
        expected_keys = ["name", "location", "rating", "price", "amenities"]

        for key in expected_keys:
            value = data.get(key) # Lấy giá trị từ JSON, trả về None nếu không có khóa

            if value is None:
                # Nếu không tìm thấy khóa hoặc giá trị là None, gán chuỗi rỗng
                extracted_data[key] = ""
            elif isinstance(value, (list, dict)):
                # Nếu giá trị là list hoặc dict, chuyển đổi thành chuỗi và strip
                extracted_data[key] = str(value).strip()
            else:
                # Nếu là kiểu dữ liệu khác (string, int, float, bool), chuyển đổi thành chuỗi và strip
                extracted_data[key] = str(value).strip()

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ JSON Parse Error: {e}")
        print(f"LLM Response (full): \n{llm_response}") # In toàn bộ phản hồi để debug
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
    writer = csv.DictWriter(out_f, fieldnames=["filename", "name", "location", "rating", "price", "amenities"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "name": "", "location": "", "rating": "", "price": "", "amenities": ""})
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
            writer.writerow({"filename": filename, "name": "", "location": "", "rating": "", "price": "", "amenities": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")