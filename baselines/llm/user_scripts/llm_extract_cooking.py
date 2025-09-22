import os
import csv
import json
import re # Thêm thư viện regex
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch # Đảm bảo import torch cho torch_dtype

# ==== CẤU HÌNH ====
HTML_DIR = "cooking_html"
CSV_FILE_LIST = "file_list3.csv"  # chứa cột "filename"
OUTPUT_CSV = "llm_output_cooking3.csv" # Đổi tên file output để tránh ghi đè

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "auto" # device_map="auto" sẽ tự động quản lý

# ==== TẢI MÔ HÌNH ====
print(f"Đang tải mô hình LLM: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Sử dụng torch_dtype=torch.bfloat16 hoặc torch.float16 nếu gặp vấn đề về bộ nhớ
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
    Cố gắng tìm phần nội dung chính của công thức.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Loại bỏ các thẻ không chứa nội dung chính hoặc gây nhiễu
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "ins", "button", "iframe", "noscript"]):
        script_or_style.decompose()

    # Cố gắng tìm phần nội dung chính của công thức
    # Các bộ chọn này cần được điều chỉnh/bổ sung dựa trên phân tích các domain của bạn
    main_content = None
    # Food Network (ví dụ: div.recipe-info, div.o-Asset)
    main_content = soup.find("div", class_="o-Asset-content") or soup.find("div", class_="recipe-info")
    # NYTimes Cooking (ví dụ: article, main)
    if not main_content:
        main_content = soup.find("article") or soup.find("main")
    # Allrecipes (ví dụ: div.recipe-content-container)
    if not main_content:
        main_content = soup.find("div", id="recipe-content-container")
    # Tasty (ví dụ: div.recipe-content) - có thể phức tạp hơn, có thể cần lấy toàn bộ body
    if not main_content:
        main_content = soup.find("div", class_="recipe-content")

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
    # Mistral-7B có context window 32k tokens. Tuy nhiên, để an toàn và hiệu quả,
    # chúng ta nên giữ dưới 4000 ký tự (khoảng 1000 tokens tiếng Anh, có thể ít hơn cho tiếng Việt)
    # hoặc điều chỉnh tùy theo kích thước trung bình của công thức và giới hạn thực tế của mô hình
    short_text_for_llm = clean_html_text[:4000]

    prompt = (
        f"[INST] Extract the following fields from the provided cooking recipe text and return them in JSON format:\n"
        f"- `name`: The name of the dish/recipe.\n"
        f"- `rating`: The numerical rating (e.g., '4.5', '5 stars'). If not available, return an empty string.\n"
        f"- `author`: The name of the person who created or published the recipe (if available). If not available, return an empty string.\n"
        f"- `time`: The total cooking or preparation time (e.g., '30 minutes', '1 hour 15 min', '2h'). If not available, return an empty string.\n"
        f"- `type`: The category or type of cuisine/dish (e.g., 'dessert', 'breakfast', 'main course', 'Italian', 'Vegetarian'). If not available, return an empty string.\n\n"
        f"Here is the recipe text:\n"
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
        return {
            "name": data.get("name", "").strip(),
            "rating": str(data.get("rating", "")).strip(), # Đảm bảo là string
            "author": data.get("author", "").strip(),
            "time": data.get("time", "").strip(),
            "type": data.get("type", "").strip()
        }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ JSON Parse Error: {e}")
        print(f"LLM Response (full): {llm_response}") # In toàn bộ phản hồi để debug
        # Trả về các trường rỗng nếu có lỗi
        return {"name": "", "rating": "", "author": "", "time": "", "type": ""}

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
    writer = csv.DictWriter(out_f, fieldnames=["filename", "name", "rating", "author", "time", "type"])
    writer.writeheader()

    for filename in tqdm(filenames, desc="🔍 Đang xử lý các HTML"):
        html_path = os.path.join(HTML_DIR, filename)
        if not os.path.exists(html_path):
            print(f"❌ File không tồn tại: {filename}")
            # Ghi một hàng trống cho file không tồn tại
            writer.writerow({"filename": filename, "name": "", "rating": "", "author": "", "time": "", "type": ""})
            continue

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # 1. Làm sạch HTML và trích xuất văn bản thô
            clean_text = get_clean_text_from_html(html_content)

            # 2. Tạo prompt cho LLM
            prompt = create_llm_prompt(clean_text)

            # 3. Gọi LLM
            # Các tham số max_new_tokens, do_sample, temperature đã được đặt trong pipeline
            llm_raw_response = pipe(prompt)[0]["generated_text"]

            # 4. Phân tích cú pháp đầu ra của LLM
            extracted_data = parse_llm_response_to_json(llm_raw_response)

            # Thêm filename và ghi vào CSV
            extracted_data["filename"] = filename
            writer.writerow(extracted_data)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")
            # Ghi một hàng trống nếu có lỗi nghiêm trọng trong quá trình xử lý file
            writer.writerow({"filename": filename, "name": "", "rating": "", "author": "", "time": "", "type": ""})

print(f"\n✅ Hoàn tất xử lý. Kết quả được lưu tại '{OUTPUT_CSV}'")