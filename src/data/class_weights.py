import os, glob, torch
from collections import Counter
import numpy as np

GRAPH_DIR = "graph_pt_output"
BIO_LABELS = [
    "O", "B-name", "I-name", "B-price", "I-price",
    "B-material", "I-material", "B-color", "I-color","B-size", "I-size"
]
label2id = {l:i for i,l in enumerate(BIO_LABELS)}

cnt = Counter()
print(f"Counting labels from .pt files in: {GRAPH_DIR}")
for f in glob.glob(os.path.join(GRAPH_DIR,"*.pt")):
    try:
        y = torch.load(f, weights_only=False).y
        cnt.update(y.tolist())
    except Exception as e:
        print(f"error load file {f}: {e}")

if not cnt:
    print("No labels found to count. Make sure the GRAPH_DIR directory contains valid .pt files.")
    exit()

# Khởi tạo mảng tần suất với 0
freq_array = np.zeros(len(BIO_LABELS))
for i in range(len(BIO_LABELS)):
    freq_array[i] = cnt[i]

total_samples = np.sum(freq_array)

# Đảm bảo total_samples không phải 0 để tránh lỗi chia cho 0
if total_samples == 0:
    print("Total number of label samples is 0. Cannot calculate weight.")
    exit()

# Tính toán trọng số theo công thức N_total / (N_class * num_classes)
# Hoặc một biến thể của inverse frequency
weights = np.zeros(len(BIO_LABELS))
num_classes = len(BIO_LABELS)

for i, label in enumerate(BIO_LABELS):
    # Tránh chia cho 0 nếu tần suất là 0
    if freq_array[i] == 0:
        # Gán một trọng số rất lớn cho các nhãn không xuất hiện
        # Hoặc một giá trị mặc định để nó không làm hỏng mean()
        # Trong trường hợp này, việc gán giá trị lớn là có ý nghĩa nếu bạn muốn mô hình học chúng
        # nhưng cần đảm bảo chúng không làm lệch hoàn toàn mean()
        weights[i] = total_samples / (1 + 1e-6) # Chia cho một số rất nhỏ để có trọng số lớn
                                                # nhưng không phải vô hạn để tránh ảnh hưởng đến mean
    else:
        # Công thức nghịch đảo tần suất lớp có chuẩn hóa
        weights[i] = total_samples / (freq_array[i] * num_classes) # Hoặc chỉ 1.0 / freq_array[i]

# Chuyển về tensor và chuẩn hóa một cách an toàn hơn
weights_tensor = torch.tensor(weights, dtype=torch.float32)

# Chuẩn hóa để trọng số của lớp 'O' là 1
# Điều này giúp giữ cho các trọng số của lớp thiểu số cao hơn đáng kể
if label2id["O"] in cnt and cnt[label2id["O"]] > 0:
    weights_tensor = weights_tensor / weights_tensor[label2id["O"]]
else:
    # Nếu lớp 'O' không tồn tại (rất hiếm), chuẩn hóa theo mean hoặc sum
    print("Warning: Class 'O' does not exist or has frequency 0. Normalize by sum.")
    weights_tensor = weights_tensor / weights_tensor.sum() * num_classes # Chuẩn hóa để tổng trọng số là num_classes

print("\n--- Label frequency ---")
for i, label in enumerate(BIO_LABELS):
    print(f"{label}: {freq_array[i]/total_samples:.6f} (Count: {int(freq_array[i])})")

print("\n--- The final weight is saved ---")
for i, label in enumerate(BIO_LABELS):
    print(f"{label}: {weights_tensor[i].item():.4f}")

torch.save(weights_tensor, "class_weights.pt")
print("\n✅ Saved class_weights.pt")
