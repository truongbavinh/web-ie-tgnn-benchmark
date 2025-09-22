import os
import json
from bs4 import BeautifulSoup
from collections import OrderedDict

def get_node_position(element, counter=None):
    """Gán node_id cho mỗi node HTML (giống như code ground truth của bạn)"""
    if counter is None:
        counter = {'count': 0}
    element.node_id = counter['count']
    counter['count'] += 1
    for child in element.children:
        if hasattr(child, 'children'):
            get_node_position(child, counter)
    return element.node_id

def tokenize_and_label(html_path, ground_truth_per_file):
    """Gán nhãn BIO cho từng token trong file HTML dựa trên ground truth"""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    get_node_position(soup)  # Make sure node_id matches ground truth

    IGNORE_TAGS = {"style", "script", "noscript", "meta", "link", "head"}

    lines = []

    for tag in soup.find_all(True):  # browse all tags
        if tag.name in IGNORE_TAGS:
            continue

        node_id = getattr(tag, "node_id", None)
        if node_id is None:
            continue

        raw_text = tag.get_text(" ", strip=True)
        if not raw_text:
            continue

        tokens = raw_text.split()
        if not tokens:
            continue

        # Find the corresponding label if node_id is present in ground truth
        str_node_id = str(node_id)
        if str_node_id in ground_truth_per_file:
            label_type = ground_truth_per_file[str_node_id]["label"]
            label_text = ground_truth_per_file[str_node_id]["text"].strip()
            gt_tokens = label_text.split()

            matched = False
            for token in tokens:
                if token in gt_tokens:
                    prefix = "B-" if not matched else "I-"
                    lines.append(f"{token}\t{prefix}{label_type}")
                    matched = True
                else:
                    lines.append(f"{token}\tO")
        else:
            for token in tokens:
                lines.append(f"{token}\tO")

    return lines

def process_all_files(html_dir, ground_truth_path, output_dir):
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for html_file in sorted(os.listdir(html_dir)):
        if not html_file.endswith(".html"):
            continue

        html_path = os.path.join(html_dir, html_file)
        base_filename = os.path.splitext(html_file)[0]
        bio_output_path = os.path.join(output_dir, f"{base_filename}.bio")

        gt_per_file = ground_truth.get(html_file, {})
        print(f"Processing {html_file}...")

        bio_lines = tokenize_and_label(html_path, gt_per_file)

        with open(bio_output_path, "w", encoding="utf-8") as out_f:
            for line in bio_lines:
                out_f.write(line + "\n")

        print(f" → Saved: {bio_output_path}")

if __name__ == "__main__":
    html_folder = "Fashion-24s"
    ground_truth_json = "ground_truth_24s.json"
    output_bio_folder = "bio_output_24s"

    process_all_files(html_folder, ground_truth_json, output_bio_folder)
