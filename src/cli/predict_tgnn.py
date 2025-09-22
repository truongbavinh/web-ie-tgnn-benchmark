import os
import re
import csv
import torch
import traceback
import unicodedata
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from sklearn.model_selection import train_test_split

# Allow loading Data objects
torch.serialization.add_safe_globals([Data])

# ==== Config ====
GRAPH_DIR = "graph_pt_output"
MODEL_PATH = "tgnn_bert_model.pt"
CSV_OUTPUT = "tgnn_predictions.csv"

LABELS = [
   "O", "B-name", "I-name",
    "B-price", "I-price",
    "B-material", "I-material",
    "B-color", "I-color",
    "B-size", "I-size"
]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
FIELDS = ["name", "price", "material", "color","size"]

# ==== Model TGNN_BERT ====
class TGNN_BERT(torch.nn.Module):
    def __init__(self, bert_hidden_size=768, hidden_dim=256,
                 num_classes=len(LABELS), num_gcn_layers=2, dropout=0.5):
        super().__init__()
        self.gcn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.gcn_layers.append(GCNConv(bert_hidden_size, hidden_dim, add_self_loops=False))
        self.batch_norms.append(BatchNorm(hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x.to(torch.float32), data.edge_index
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.gcn_layers) - 1:
                x = self.dropout(x)
        return self.classifier(x)

# ==== Function to extract labels from BIO ====
def extract_field_from_prediction(nodes, labels, field_name):
    segments = []
    current = []
    for text, label in zip(nodes, labels):
        if label == f"B-{field_name}":
            if current:
                segments.append(" ".join(current))
                current = []
            current = [text]
        elif label == f"I-{field_name}" and current:
            current.append(text)
        else:
            if current:
                segments.append(" ".join(current))
                current = []
    if current:
        segments.append(" ".join(current))
    return " | ".join(segments)

# ==== Text cleaning function ====
def clean_field_value(text: str, label: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # chuẩn hoá khoảng trắng

    parts = [p.strip() for p in re.split(r"\s*\|\s*|\n", text) if p.strip()]
    seen = set()
    clean_parts = []

    for p in parts:
        p = re.sub(r"([A-Za-z]{2,})( \1)+", r"\1", p)  # xoá từ lặp
        p_lower = p.lower()
        if p_lower not in seen:
            seen.add(p_lower)
            clean_parts.append(p)

    blocked_kw = [
        "start trial", "learn more", "buy for", "included with", "free trial",
        "rating", "ratings", "watch now", "current price", "students",
        "liked by", "skill level", "report", "source", "released", "enroll"
    ]
    clean_parts = [p for p in clean_parts if not any(kw in p.lower() for kw in blocked_kw)]

    max_len = {
        "name": 150,
        "price": 40,
        "material": 200,
        "color": 100,
        "size": 100
    }.get(label, 100)

    if label == "name":
        candidates = [p for p in clean_parts if len(p) >= 10]
        if candidates:
            best = candidates[0]
            if len(best) > max_len:
                best = best[:max_len].rsplit(" ", 1)[0] + "..."
            return best
        return text[:max_len]

    elif label in {"price", "material", "color", "size"}:
        return " | ".join(clean_parts[:3])[:max_len]

    else:
        return clean_parts[0][:max_len] if clean_parts else text[:max_len]
# ==== Prediction function for 1 file ====
def predict_file(model, fpath):
    data = torch.load(fpath, weights_only=False)
    data.x = data.x.float()
    data = data.to(device)
    out = model(data)
    pred = out.argmax(dim=1).tolist()
    labels = [ID2LABEL[i] for i in pred]

    result = {"filename": os.path.basename(fpath).replace(".pt", "")}
    for field in FIELDS:
        raw = extract_field_from_prediction(data.nodes, labels, field)
        clean = clean_field_value(raw, field)
        result[field] = clean
    return result

# ==== Load model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TGNN_BERT().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==== Chia dữ liệu ====
all_files = sorted(f for f in os.listdir(GRAPH_DIR) if f.endswith(".pt"))
train_val, test_files = train_test_split(all_files, test_size=0.15,random_state=42)
train_files, val_files = train_test_split(train_val, test_size=0.1765,random_state=42)

print(f"📊 Tổng file: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# ==== Dự đoán tập test ====
rows = []
for fname in tqdm(test_files, desc="🔍 Predicting test set"):
    fpath = os.path.join(GRAPH_DIR, fname)
    try:
        result = predict_file(model, fpath)
        rows.append(result)
    except Exception as e:
        print(f"\n⚠️ Lỗi {fname}: {e}")
        print(traceback.format_exc())

# ==== Ghi ra CSV ====
with open(CSV_OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename"] + FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Đã lưu kết quả dự đoán test vào: {CSV_OUTPUT}")

