import os, numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# ==== CONFIG ====
BIO_DIR = "bio_output"                  # Folder contain file .bio
MODEL_NAME = "bert-base-uncased"       # BERT used for training
MAX_LEN = 256

# ==== READ FILE .BIO ====
def read_bio_folder(bio_dir):
    sents, labels = [], []
    for fname in os.listdir(bio_dir):
        if not fname.endswith(".bio"):
            continue
        with open(os.path.join(bio_dir, fname), encoding="utf-8") as f:
            tok_lst, lab_lst = [], []
            for line in f:
                if not line.strip():
                    if tok_lst:
                        sents.append(tok_lst); labels.append(lab_lst)
                        tok_lst, lab_lst = [], []
                    continue
                parts = line.strip().split()
                if len(parts) != 2:
                    continue  # bỏ dòng lỗi
                tok, lab = parts
                tok_lst.append(tok); lab_lst.append(lab)
            if tok_lst:
                sents.append(tok_lst); labels.append(lab_lst)
    return sents, labels

tokens_list, labels_list = read_bio_folder(BIO_DIR)

# ==== GÁN label2id / id2label ====
unique_labels = sorted({lab for labs in labels_list for lab in labs})
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}

# ==== Chia train/val ====
def to_dataset(tokens, labels):
    return Dataset.from_dict({
        "tokens": tokens,
        "ner_tags": [[label2id[l] for l in labs] for labs in labels]
    })

split_idx = int(len(tokens_list) * 0.9)
ds = DatasetDict({
    "train": to_dataset(tokens_list[:split_idx], labels_list[:split_idx]),
    "validation": to_dataset(tokens_list[split_idx:], labels_list[split_idx:]),
})

# ==== Tokenize và align label ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        max_length=MAX_LEN,
        is_split_into_words=True,
    )
    word_ids = tokenized.word_ids()
    aligned = []
    prev_word = None
    for w_id in word_ids:
        if w_id is None:
            aligned.append(-100)
        elif w_id != prev_word:
            aligned.append(example["ner_tags"][w_id])
        else:
            aligned.append(-100)
        prev_word = w_id
    tokenized["labels"] = aligned
    return tokenized

ds = ds.map(tokenize_and_align, batched=False)

# ==== MODEL + COLLATOR ====
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
data_collator = DataCollatorForTokenClassification(tokenizer)

# ==== EVALUATION ====
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for lrow, prow in zip(labels, preds):
        for l, p in zip(lrow, prow):
            if l != -100:
                true_labels.append(id2label[l])
                true_preds.append(id2label[p])
    return metric.compute(predictions=[true_preds], references=[true_labels])

# ==== TRAINING ARGUMENTS ====
args = TrainingArguments(
    output_dir="bert_ner_ckpt",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,  # sẽ dừng sớm nếu không cải thiện
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True
)

# ==== TRAINER ====
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ==== TRAIN ====
trainer.train()
trainer.save_model("bert_ner_final")
tokenizer.save_pretrained("bert_ner_final")
