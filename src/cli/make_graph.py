import os
import json
import torch
import re
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from collections import Counter

# ========================== Config ==========================
HTML_DIR = 'fashion_html'  # folder contain html file
GROUND_TRUTH_PATH = 'ground_truth/ground_truth_24s.json'  # Ground truth file path
OUTPUT_DIR = 'graph_pt_output'

BERT_PATH = 'bert_ner_final'

BIO_LABELS = [
    "O", "B-name", "I-name",
    "B-price", "I-price",
    "B-material", "I-material",
    "B-color", "I-color",
    "B-size", "I-size"
]
LABEL2IDX = {label: idx for idx, label in enumerate(BIO_LABELS)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================== Load BERT ==========================
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert_model = BertModel.from_pretrained(BERT_PATH).to(device)

# ========================== Ground truth ==========================
with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# ========================== Helper ==========================
def get_node_position(element, counter=None):
    if counter is None:
        counter = {'count': 0}
    element.node_id = str(counter['count'])
    counter['count'] += 1
    for child in element.children:
        if hasattr(child, 'children'):
            get_node_position(child, counter)
    return element.node_id

def get_node_features(nodes, tokenizer, model, device, batch_size=32):
    features = []
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        features.append(cls_embeddings)
        torch.cuda.empty_cache()
    return torch.cat(features, dim=0)

# ========================== Process HTML ==========================
def process_html_file(file_path):
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.pt")

    if os.path.exists(output_path):
        print(f"♻️  overwriting: {file_id}.pt (existed)")
        # không return, sẽ ghi đè

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except Exception as e:
        print(f" Filter HTML {file_path}: {e}")
        return

    if not soup or not soup.body:
        print(f" Invalid HTML: {file_path}")
        return

    get_node_position(soup)
    node_map = {tag.node_id: tag for tag in soup.body.descendants if hasattr(tag, 'node_id')}
    sorted_nodes = sorted(node_map.items(), key=lambda x: int(x[0]))
    node_texts, node_ids = [], []

    for node_id, tag in sorted_nodes:
        text = tag.get_text(separator=' ', strip=True)
        if text:
            node_texts.append(text)
            node_ids.append(node_id)

    num_nodes = len(node_texts)
    if num_nodes == 0:
        print(f" No text node found in {file_path}")
        return

    y = torch.full((num_nodes,), LABEL2IDX["O"], dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)

    gt_nodes = ground_truth.get(file_id + ".html", {})

    # ✅ Gán nhãn từ ground truth bằng node_id
    used_label_count = {}
    for idx, node_id in enumerate(node_ids):
        if node_id in gt_nodes:
            label = gt_nodes[node_id]["label"]
            tag = "B-" + label if used_label_count.get(label, 0) == 0 else "I-" + label
            y[idx] = LABEL2IDX.get(tag, LABEL2IDX["O"])
            train_mask[idx] = True
            used_label_count[label] = used_label_count.get(label, 0) + 1

    # ==== BERT Embedding ====
    try:
        x = get_node_features(node_texts, tokenizer, bert_model, device)
        x = x.to(torch.float16)
    except Exception as e:
        print(f"GPU error → switch CPU: {file_id}: {e}")
        bert_model.to('cpu')
        x = get_node_features(node_texts, tokenizer, bert_model, 'cpu')
        x = x.to(torch.float16)

    # ==== Edge index ====
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)

    # ==== Tạo Data object ====
    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask)
    data.nodes = node_texts
    data.node_ids = node_ids

    torch.save(data, output_path)
    label_count = Counter(y.tolist())
    summary = {BIO_LABELS[i]: label_count[i] for i in range(len(BIO_LABELS)) if label_count[i] > 0}

    print(f"{file_id}.pt | Nodes: {num_nodes} | Labeled: {train_mask.sum().item()}")
    print(f" Label: {summary}")

# ========================== Run ==========================
all_html_files = sorted([
    os.path.join(HTML_DIR, f)
    for f in os.listdir(HTML_DIR)
    if f.endswith(".html") and os.path.splitext(f)[0] in {os.path.splitext(fn)[0] for fn in ground_truth}
])

print(f"Total number of HTML files to process: {len(all_html_files)}")

for html_file in all_html_files:
    process_html_file(html_file)
