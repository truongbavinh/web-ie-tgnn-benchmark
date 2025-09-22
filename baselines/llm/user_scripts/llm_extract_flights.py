import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "flights_html" # Đảm bảo thư mục này chứa các file HTML chuyến bay của bạn
CSV_FILE_LIST = "file_list2.csv" # Đảm bảo file này chứa cột "filename" với tên các file HTML
OUTPUT_CSV = "llm_output_flights2.csv" # Đổi tên file output để tránh ghi đè

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
    Cố gắng tìm phần nội dung chính của thông tin chuyến bay.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ các thẻ không chứa nội dung chính hoặc gây nhiễu
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript", "meta[name='description']"]):
        script_or_style.decompose()

    main_content = None

    # Các bộ chọn tiềm năng cho các trang chuyến bay.
    # Thêm các bộ chọn phổ biến từ các trang như Kayak, CheapOair, Gotogate, FareCompare, TripAdvisor.
    # Đây là danh sách tổng hợp, bạn có thể cần kiểm tra và tinh chỉnh thêm nếu gặp trang mới.
    selectors = [
        "div.main.clearfix", # generic, but sometimes useful
        "div.FlightItineraryDetails__details__fc", # from some flight sites
        "div.LoWCY", # generic content wrapper
        "main", # HTML5 main content tag
        "div.OQa--content", # generic content wrapper
        "div.ReactModal__Content", # often contains pop-up details
        "div#root", # common root element for React/SPA apps
        "div.results-summary",
        "div.results-page",
        "div.flight-details",
        "section.flight-info",
        "div.booking-details",
        "div.trip-details",
        "div.fare-details",
        "div.flight-card", # common for individual flight listings
        "div.booking-detail-item",
        "div.itinerary-section",
        "div.content-wrapper",
        "div.flight-info-container",
        "div.booking-details-wrapper", # Specific to some booking sites
        "div.details-segment", # Often contains segment details
        "div.flight-summary", # Summaries
        "div[data-component='flight-results-page']", # Kayak specific
        "div.section.module.itinerary", # TripAdvisor
        "div.section.module.flight-options", # TripAdvisor
        "div.flight-result-details", # General result details
        "div.leg-details", # For individual flight legs
        "div.flex-container", # Sometimes used for layout of flight info
        "div.travel-information"
    ]

    for selector in selectors:
        main_content = soup.find(class_=selector.split('.')[-1]) if '.' in selector else soup.find(id=selector.split('#')[-1]) if '#' in selector else soup.find(selector)
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
    short_text_for_llm = clean_html_text[:4000] # Giữ 4000 ký tự để an toàn, có thể điều chỉnh

    prompt = (
        f"[INST] Extract the following fields from the provided flight information text and return them in JSON format. Ensure all values are plain strings.\n"
        f"- `name`: The flight route or general name of the trip (e.g., 'London to New York', 'Round-trip flight', 'Chuyến bay Hà Nội đi TP.HCM'). Return an empty string if not available.\n"
        f"- `duration`: The total duration of the flight, including layovers (e.g., '10 hours 30 minutes', '2h 15m', '12:45'). If there are multiple durations for different legs, provide them as a comma-separated string (e.g., '2h 30m, 5h 10m'). Return an empty string if not available.\n"
        f"- `stop`: The number of stops or layovers (e.g., '0 stops', '1 stop', 'direct', '2'). Return an empty string if not available.\n"
        f"- `price`: The total price of the flight, including currency (e.g., '$500', '1,234,567 VND', '£150'). Return an empty string if not available.\n"
        f"- `departure_time`: The departure time of the flight, including date if available (e.g., '10:00 AM', 'Sat, Jun 14 8:40 AM', '15:20 25/07'). If multiple legs, provide as a comma-separated string. Return an empty string if not available.\n"
        f"- `arrival_time`: The arrival time of the flight, including date if available (e.g., '02:30 PM', 'Sun, Jun 15 10:20 AM', '18:55 25/07'). If multiple legs, provide as a comma-separated string. Return an empty string if not available.\n"
        f"- `airline`: The airline company or companies operating the flight (e.g., 'Vietnam Airlines', 'Emirates, Qatar Airways'). If multiple airlines, provide as a comma-separated string. Return an empty string if not available.\n\n"
        f"Here is the flight text:\n"
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
        "duration": "",
        "stop": "",
        "price": "",
        "departure_time": "",
        "arrival_time": "",
        "airline": ""
    }

    try:
        # Tìm phần JSON trong phản hồi của LLM, có thể được bọc trong ```json ... ```
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: nếu không có ```json, tìm dấu { đầu tiên và } cuối cùng
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = llm_response[json_start : json_end + 1]
            else:
                raise ValueError("No valid JSON structure found in LLM response.")

        data = json.loads(json_str)

        # Danh sách các khóa mong muốn để đảm bảo xử lý nhất quán
        expected_keys = ["name", "duration", "stop", "price", "departure_time", "arrival_time", "airline"]

        for key in expected_keys:
            value = data.get(key) # Lấy giá trị từ JSON, trả về None nếu không có khóa

            if value is None:
                extracted_data[key] = "" # Gán chuỗi rỗng nếu giá trị là None
            elif isinstance(value, (list, dict)):
                # Nếu giá trị là list hoặc dict, chuyển đổi thành chuỗi và strip
                extracted_data[key] = str(value).strip()
            else:
                # Nếu là kiểu dữ liệu khác (string, int, float, bool), chuyển đổi thành chuỗi và strip
                extracted_data[key] = str(value).strip()

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Lỗi phân tích JSON: {e}")
        print(f"LLM Response (full): {llm_response}") # In toàn bộ phản hồi để debug
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
# Mở file output ở chế độ 'w' (write) để đảm bảo ghi từ đầu
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["filename", "name","duration", "stop" , "price", "departure_time", "arrival_time", "airline"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            # Ghi một hàng trống cho file không tồn tại
            writer.writerow({"filename": filename, "name": "", "duration": "", "stop": "", "price": "", "departure_time": "", "arrival_time": "", "airline": ""})
            continue

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # 1. Làm sạch HTML và trích xuất văn bản thô
            clean_text = get_clean_text_from_html(html_content)

            # 2. Tạo prompt cho LLM
            prompt = create_llm_prompt(clean_text)

            # 3. Gọi LLM
            llm_raw_response = pipe(prompt)[0]["generated_text"]

            # 4. Phân tích cú pháp đầu ra của LLM
            extracted_data = parse_llm_response_to_json(llm_raw_response)

            # Thêm filename và ghi vào CSV
            extracted_data["filename"] = filename
            writer.writerow(extracted_data)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            # Ghi một hàng trống nếu có lỗi nghiêm trọng trong quá trình xử lý file
            writer.writerow({"filename": filename, "name": "", "duration": "", "stop": "", "price": "", "departure_time": "", "arrival_time": "", "airline": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")