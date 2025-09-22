import argparse, json, os, pandas as pd, glob

def main():
    ap = argparse.ArgumentParser(description="Aggregate results/*/summary.json to leaderboard.csv")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="leaderboard.csv")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.results_dir, "**", "summary.json"), recursive=True):
        team = os.path.basename(os.path.dirname(path))
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        row = {"team_or_method": team, "task_name": js.get("task_name","unknown")}
        for k,v in js.get("metrics", {}).items():
            row[k] = v
        rows.append(row)

    if not rows:
        print("No results found.")
        return
    df = pd.DataFrame(rows)
    # sort by accuracy desc if present
    sort_cols = [c for c in ["accuracy","f1_macro"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False]*len(sort_cols))
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
