# -*- coding: utf-8 -*-
"""
BERT pipeline for token classification (BIO tags).
Interface:
    train(train_path: str, val_path: Optional[str], ckpt_out: str, seed: int = 42)
Assumptions about train/val .jsonl:
    Each line:
    {
      "id": "fashion-0001",               # optional
      "domain": "fashion",                # optional
      "tokens": ["Striped", "Shorts", ...],
      "tags":   ["B-NAME", "I-NAME", ...] # BIO tags (same length as tokens)
    }
This saves a HF checkpoint folder at `ckpt_out`.
"""

from typing import List, Dict, Optional
import json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          set_seed)

# ----------------------------
# Data utils
# ----------------------------
def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def _build_label_list(examples: List[Dict]) -> List[str]:
    labels = set()
    for ex in examples:
        for t in ex["tags"]:
            labels.add(t)
    label_list = sorted(labels)
    return label_list

class BioDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, label2id: Dict[str,int], max_len: int = 512):
        self.data = data
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        tokens: List[str] = ex["tokens"]
        tags:   List[str] = ex["tags"]
        enc = self.tok(tokens,
                       is_split_into_words=True,
                       truncation=True,
                       max_length=self.max_len,
                       return_tensors="pt",
                       return_offsets_mapping=False)
        # Align labels to wordpiece
        word_ids = enc.word_ids(batch_index=0)  # type: ignore
        label_ids = []
        last_word_id = None
        for wi in word_ids:
            if wi is None:
                label_ids.append(-100)  # ignore
            else:
                tag = tags[wi]
                # Only label first piece, set subsequent pieces to -100 (common practice)
                if wi != last_word_id:
                    label_ids.append(self.label2id[tag])
                    last_word_id = wi
                else:
                    label_ids.append(-100)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return item

# ----------------------------
# Public API
# ----------------------------
def train(train_path: str, val_path: Optional[str], ckpt_out: str, seed: int = 42):
    os.makedirs(ckpt_out, exist_ok=True)
    set_seed(seed)

    # ---- Load data
    train_data = _read_jsonl(train_path)
    val_data   = _read_jsonl(val_path) if val_path else None

    # ---- Build labels
    label_list = _build_label_list(train_data + (val_data or []))
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}

    # ---- Model & tokenizer
    model_name = os.environ.get("BERT_MODEL_NAME", "bert-base-uncased")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # ---- Datasets
    train_ds = BioDataset(train_data, tok, label2id)
    eval_ds = BioDataset(val_data, tok, label2id) if val_data else None

    data_collator = DataCollatorForTokenClassification(tok)

    # ---- Training args
    args = TrainingArguments(
        output_dir=ckpt_out,
        per_device_train_batch_size=int(os.environ.get("BERT_BS", 8)),
        per_device_eval_batch_size=int(os.environ.get("BERT_EVAL_BS", 8)),
        learning_rate=float(os.environ.get("BERT_LR", 5e-5)),
        num_train_epochs=float(os.environ.get("BERT_EPOCHS", 3)),
        logging_steps=50,
        evaluation_strategy="steps" if eval_ds else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if eval_ds else False,
        report_to=[],
        seed=seed,
    )

    def _metrics_fn(p):
        # optional: compute token-level micro-F1 over non -100 labels
        from sklearn.metrics import f1_score
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids
        y_true, y_pred = [], []
        for lt, pt in zip(labels, preds):
            for l, pr in zip(lt, pt):
                if l == -100: continue
                y_true.append(l)
                y_pred.append(pr)
        f1 = f1_score(y_true, y_pred, average="micro") if y_true else 0.0
        return {"token_f1_micro": f1}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=_metrics_fn if eval_ds else None,
    )

    trainer.train()
    # Save final artifacts
    trainer.save_model(ckpt_out)
    tok.save_pretrained(ckpt_out)
    # Persist label maps for downstream TGNN
    with open(os.path.join(ckpt_out, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

