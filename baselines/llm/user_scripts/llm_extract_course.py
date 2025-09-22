import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "course_html" # Đảm bảo thư mục này chứa các file HTML về khóa học của bạn
CSV_FILE_LIST = "file_list3.csv" # Đảm bảo file này chứa cột "filename" với tên các file HTML
OUTPUT_CSV = "llm_output_course3.csv" # Đổi tên file output để tránh ghi đè

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
    Cố gắng tìm phần nội dung chính của thông tin khóa học.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ các thẻ không chứa nội dung chính hoặc gây nhiễu
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript", "meta[name='description']"]):
        script_or_style.decompose()

    main_content = None

    # Các bộ chọn tiềm năng cho các trang khóa học.
    # Đây là danh sách tổng hợp, bạn có thể cần kiểm tra và tinh chỉnh thêm
    # dựa trên cấu trúc HTML của các trang web cụ thể mà bạn đang làm việc.
    selectors = [
        "div.course-detail", # Phổ biến cho trang chi tiết khóa học
        "div.course-info",
        "div.course-description",
        "section.course-overview",
        "div.main-content",
        "article.course-page",
        "div#course-main-content",
        "div.col-md-8.course-content", # Ví dụ từ cấu trúc chia cột
        "div.wrapper", # Bộ chọn chung, đôi khi hữu ích
        "div.container-fluid.ContentWrapper", # Bộ chọn chung khác
        "div[data-component='course-details']", # Các trang dùng framework
        "div.css-xxxxxx", # Các lớp CSS tự động sinh ra có thể cần phân tích
        "div.instructor-details", # Thông tin giảng viên
        "div.course-fees", # Thông tin học phí
        "div.course-duration" # Thông tin thời lượng
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
        f"[INST] Extract the following fields from the provided course information text and return them in JSON format. Ensure all values are plain strings.\n"
        f"- `title`: The title or name of the course (e.g., 'Introduction to Python Programming', 'Advanced Data Science', 'Graphic Design Fundamentals'). Return an empty string if not available.\n"
        f"- `subject`: The main subject or category of the course (e.g., 'Computer Science', 'Business', 'Arts & Humanities', 'Marketing'). Return an empty string if not available.\n"
        f"- `duration`: The total duration of the course (e.g., '8 weeks', '3 months', '40 hours', 'Self-paced'). Return an empty string if not available.\n"
        f"- `fees`: The cost or tuition fees for the course, including currency (e.g., '$99.99', '5,000,000 VND', 'Free'). Return an empty string if not available.\n"
        f"- `instructor`: The name(s) of the instructor(s) teaching the course (e.g., 'Dr. Jane Doe', 'Prof. John Smith', 'Various'). If multiple instructors, provide as a comma-separated string. Return an empty string if not available.\n\n"
        f"Here is the course text:\n"
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
        "title": "",
        "subject": "",
        "duration": "",
        "fees": "",
        "instructor": ""
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
        expected_keys = ["title", "subject", "duration", "fees", "instructor"]

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
    writer = csv.DictWriter(out_f, fieldnames=["filename", "title", "subject", "duration", "fees", "instructor"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            # Ghi một hàng trống cho file không tồn tại
            writer.writerow({"filename": filename, "title": "", "subject": "", "duration": "", "fees": "", "instructor": ""})
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
            writer.writerow({"filename": filename, "title": "", "subject": "", "duration": "", "fees": "", "instructor": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")