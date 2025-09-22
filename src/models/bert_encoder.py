# -*- coding: utf-8 -*-
"""
Lightweight BERT encoder / token-classification wrapper.
"""
from typing import List
import os, json, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BertTokenEncoder:
    def __init__(self, ckpt_dir_or_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir_or_name)
        self.model = AutoModelForTokenClassification.from_pretrained(ckpt_dir_or_name)
        self.label2id = getattr(self.model.config, "label2id", None)
        self.id2label = getattr(self.model.config, "id2label", None)
        l2i_path = os.path.join(ckpt_dir_or_name, "label2id.json")
        if os.path.exists(l2i_path):
            with open(l2i_path, "r", encoding="utf-8") as f:
                self.label2id = json.load(f)
            self.id2label = {int(v): k for k, v in self.label2id.items()}
        self.model.eval()

    @torch.no_grad()
    def predict_tags(self, tokens: List[str], max_length: int = 512) -> List[str]:
        enc = self.tokenizer(tokens, is_split_into_words=True,
                             truncation=True, max_length=max_length,
                             return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0)
        logits = self.model(**{k: v for k, v in enc.items()}).logits[0]
        pred_ids = logits.argmax(-1).tolist()
        tags, last = [], None
        for w, pid in zip(word_ids, pred_ids):
            if w is None: continue
            if w != last:
                tag = self.id2label.get(pid, str(pid)) if self.id2label else str(pid)
                tags.append(tag); last = w
        return tags

    @torch.no_grad()
    def encode(self, tokens: List[str], max_length: int = 512):
        enc = self.tokenizer(tokens, is_split_into_words=True,
                             truncation=True, max_length=max_length,
                             return_tensors="pt")
        out = self.model.bert(**{k: v for k, v in enc.items()})  # type: ignore
        return out.last_hidden_state[0]  # [T, H]
