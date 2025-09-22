import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
import torch.serialization

# === Allow loading .pt files from PyG ===
torch.serialization.add_safe_globals([Data])

# ==== config domain ====
GRAPH_DIR = "graph_pt_output"
MODEL_PATH = "tgnn_bert_model.pt"
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
SEED = 42

# ==== Labels used in the domain ====
LABELS = [
    "O", "B-name", "I-name",
    "B-price", "I-price",
    "B-material", "I-material",
    "B-color", "I-color",
    "B-size", "I-size"
]
label2id = {l: i for i, l in enumerate(LABELS)}

# ==== Set seed to reproduce results ====
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==== Definition of TGNN_BERT model ====
class TGNN_BERT(nn.Module):
    def __init__(self, bert_hidden_size=768, hidden_dim=256,
                 num_classes=len(LABELS), num_gcn_layers=2, dropout=0.5):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.gcn_layers.append(GCNConv(bert_hidden_size, hidden_dim, add_self_loops=False))
        self.batch_norms.append(BatchNorm(hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x.to(torch.float32), data.edge_index
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.gcn_layers) - 1:
                x = self.dropout(x)
        return self.classifier(x)

# ==== Load danh sách file .pt ====
all_files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.pt")))
print(f"\n Find {len(all_files)} file .pt in folder: {GRAPH_DIR}")
if not all_files:
    print("pt file not found. Stop.")
    exit()

# ==== Load dữ liệu từng file .pt (ép float32 để tránh lỗi) ====
data_list = []
print("Loading each file .pt...")
for f in tqdm(all_files):
    try:
        data = torch.load(f, weights_only=False)
        data.x = data.x.to(torch.float32)
        data.edge_index = data.edge_index.contiguous().to(torch.long)
        data_list.append(data)
    except Exception as e:
        print(f"Error load {f}: {e}")

if not data_list:
    print("No files loaded. Stop.")
    exit()

# ==== Đếm nhãn để tính class weights ====
weights_path = "class_weights.pt"
if not os.path.exists(weights_path):
    raise FileNotFoundError("class_weights.pt file not found — run the weights calculation script first.")
class_weights = torch.load(weights_path).to(torch.float32)
print("Loaded class_weights from:", weights_path)

# ==== Chia train/val/test ====
n_total = len(data_list)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_data, val_data, test_data = random_split(
    data_list, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ==== Khởi tạo model và training ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TGNN_BERT(bert_hidden_size=768).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

best_val_loss = float("inf")

print("\n🚀 Begin training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y)
            val_loss += loss.item()

    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"📈 Epoch {epoch:02d} | Train loss: {avg_train:.4f} | Val loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), MODEL_PATH)
        print(" Best model saved.")

print("\nTraining complete. Best model saved at:", MODEL_PATH)
