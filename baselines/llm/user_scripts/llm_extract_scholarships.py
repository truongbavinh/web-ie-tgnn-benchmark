import os
import csv
import json
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ==== CẤU HÌNH ====
HTML_DIR = "scholarships_html"
CSV_FILE_LIST = "file_list3.csv"
OUTPUT_CSV = "llm_output_scholarships3.csv"

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
    Cố gắng tìm phần nội dung chính của thông tin học bổng.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ các thẻ không chứa nội dung chính hoặc gây nhiễu
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript", "meta[name='description']"]): # Thêm meta description vào đây nếu bạn muốn nó bị bỏ qua trong text thô, nhưng LLM sẽ thấy nó trong prompt
        script_or_style.decompose()

    main_content = None

    # Các bộ chọn tiềm năng cho các trang học bổng
    # Cố gắng tìm các thẻ chứa nội dung chính xác hơn
    main_content = soup.find("article", class_="scholarship-details") \
                   or soup.find("section", class_="content") \
                   or soup.find("div", class_="scholarship-overview") \
                   or soup.find("div", class_="program-details") \
                   or soup.find("div", class_="scholarship-data") \
                   or soup.find("div", class_="cb-main-content") \
                   or soup.find("div", id="main-content") \
                   or soup.find("div", class_="content-area") \
                   or soup.find("div", class_="col-sm-8") # Selector phổ biến từ Internationalscholarships-0008.html

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
    short_text_for_llm = clean_html_text[:4000]

    prompt = (
        f"[INST] Extract the following fields from the provided scholarship information text and return them in JSON format. Ensure all values are plain strings.\n"
        f"- `title`: The full title or name of the scholarship.\n"
        f"- `provider`: The organization, university, or entity offering the scholarship.\n"
        f"- `amount`: The financial value of the scholarship (e.g., '$1,000', 'full tuition', 'stipend'). Return an empty string if not available.\n"
        f"- `deadline`: The application deadline for the scholarship, including month, day, and year (e.g., 'December 1, 2025', 'April 23rd'). Return an empty string if not available.\n"
        f"- `award`: Any specific award details or benefits beyond the main amount (e.g., 'monthly stipend', 'travel allowance', 'research grant'). Return an empty string if not available.\n\n"
        f"Here is the scholarship text:\n"
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
    # Khởi tạo dictionary với tất cả các khóa và giá trị mặc định là chuỗi rỗng
    extracted_data = {
        "title": "",
        "provider": "",
        "amount": "",
        "deadline": "",
        "award": ""
    }

    try:
        # Tìm phần JSON trong phản hồi của LLM
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = llm_response[json_start : json_end + 1]
            else:
                # Nếu không tìm thấy JSON hợp lệ, ghi log và trả về extracted_data với các giá trị mặc định
                print(f"⚠️ Không tìm thấy cấu trúc JSON hợp lệ trong phản hồi LLM.")
                print(f"LLM Response (full): \n{llm_response}")
                return extracted_data

        data = json.loads(json_str)

        # Danh sách các khóa mong muốn để đảm bảo xử lý nhất quán
        expected_keys = ["title", "provider", "amount", "deadline", "award"]

        for key in expected_keys:
            value = data.get(key)

            if value is None:
                extracted_data[key] = ""
            elif isinstance(value, (list, dict)):
                extracted_data[key] = str(value).strip()
            else:
                extracted_data[key] = str(value).strip()

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Lỗi phân tích JSON: {e}")
        print(f"LLM Response (full): \n{llm_response}")
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
    writer = csv.DictWriter(out_f, fieldnames=["filename", "title", "provider", "amount", "deadline", "award"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            writer.writerow({"filename": filename, "title": "", "provider": "", "amount": "", "deadline": "", "award": ""})
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
            writer.writerow({"filename": filename, "title": "", "provider": "", "amount": "", "deadline": "", "award": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")