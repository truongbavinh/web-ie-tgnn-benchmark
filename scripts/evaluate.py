import argparse, json, os, yaml
import pandas as pd
from metrics import compute
from src.metrics.slot_f1 import f1_slot_micro, f1_slot_macro
from src.metrics.exact_match_record import exact_match_record
from src.metrics.price_errors import price_mae, price_mape

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions against gold labels.")
    ap.add_argument("--gold", required=True, help="Path to gold.csv with columns: id,label")
    ap.add_argument("--pred", required=True, help="Path to predictions.csv with columns: id,prediction")
    ap.add_argument("--task", required=True, help="Path to tasks.yaml")
    ap.add_argument("--out", required=True, help="Path to write summary.json")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    pred = pd.read_csv(args.pred)

    # basic validation
    if not set(["id","label"]).issubset(gold.columns):
        raise SystemExit("gold.csv must contain columns: id,label")
    if not set(["id","prediction"]).issubset(pred.columns):
        raise SystemExit("predictions.csv must contain columns: id,prediction")

    df = gold.merge(pred, on="id", how="left", validate="one_to_one")
    missing = df["prediction"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} predictions are missing; they will be ignored.")
        df = df.dropna(subset=["prediction"])

    y_true = df["label"].values
    y_pred = df["prediction"].values

    task = load_yaml(args.task)
    metrics = task.get("metrics", [])
    results = {"task_name": task.get("task_name","unknown"), "metrics": {}}
    for m in metrics:
        name = m["name"] if isinstance(m, dict) else m
        metric_name, value = compute(name, y_true, y_pred)
        results["metrics"][metric_name] = value

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
