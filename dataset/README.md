# Datasets — Raw HTML Storage (Hosted Externally)

The **raw HTML files are *not* stored in this repository** due to size (hundreds of MB to many GB).  
They are hosted on **Hugging Face Datasets** and should be downloaded on demand.

> Dataset hub: **https://huggingface.co/datasets/vinhvinhit/web-ie-raw-html**

This README explains the layout we expect locally, and how to fetch the zipped HTML packs for each domain.

---

## 1) Local Layout

We keep zipped HTML per domain under `datasets/raw_html_zips/`:
```
datasets/
└─ swde/
   ├─ raw_html_zips/
   │  ├─ app.zip
   │  ├─ cooking.zip
   │  ├─ course.zip
   │  ├─ events.zip
   │  ├─ fashion.zip
   │  ├─ fashion.zip
   │  ├─ hotel.zip
   │  ├─ realestate.zip
   │  ├─ scholarships.zip
   │  └─ tourist.zip
   │  
   ├─ ground_truth/
   │  └─ <domain>/*.json          # per-site GT files (not from HF hub)
   └─ html/                        # (optional) temporary unzip for debugging
```

Keep the HTML **zipped** unless your pipeline truly requires an unzip step.  
All provided scripts (BIO labeling, graph building) can be written to stream directly from zip to reduce file count/IO.

---

## 2) Download from Hugging Face

You can download each domain’s zip via our helper script or the Hugging Face CLI.

### A) Python helper (recommended)
```bash
# Example: SWDE auto
python scripts/fetch_html_from_hf.py   --repo-id vinhvinhit/web-ie-raw-html   --path-in-repo swde/app.zip   --out datasets/raw_html_zips/app.zip
```
Repeat for other domains by changing `--path-in-repo` and `--out`.

### B) Hugging Face CLI (optional)
First login (one-time):
```bash
huggingface-cli login
# paste your HF access token when prompted
```

Then download any file from the dataset hub release assets:
```bash
# Example: using 'hf_hub_download' from Python REPL
python -c "from huggingface_hub import hf_hub_download as d; print(d('vinhvinhit/web-ie-raw-html', filename='app.zip', repo_type='dataset'))"
```

> If your environment is headless (e.g., server), pass the token via env var:  
> `export HUGGINGFACEHUB_API_TOKEN=<your_token>`

---

## 3) Integrity & Reproducibility

We recommend verifying checksums after download:
```bash
# Linux/macOS
sha1sum datasets/raw_html_zips/app.zip
# Windows PowerShell
Get-FileHash datasets\raw_html_zips\app.zip -Algorithm SHA1
```

If you maintain **`artifacts.yaml`**, you can record the expected `sha1` for each file:
```yaml
artifacts:
  swde_auto_zip:
    target_dir: datasets/raw_html_zips
    type: file
    url: https://huggingface.co/datasets/vinhvinhit/web-ie-raw-html/blob/main/app_html.zip
    sha1: <PUT-SHA1>
```

(Optionally) implement a small downloader that checks SHA-1 before marking the artifact “ready”.

---

## 4) Domains

For SWDE we host 8 domains (10 sites each):
- **auto** — model, price, engine, fuel  
- **book** — title, author, isbn13, pub, date  
- **camera** — model, price, manufacturer  
- **job** — title, company, location, date  
- **movie** — title, director, genre, mpaa  
- **nbaplayer** — name, team, height, weight  
- **restaurant** — name, address, phone, cuisine  
- **university** — name, phone, website, type  

Check `benchmarks/web_ie_multidomain/<domain>/README.md` and `schema.md` for domain-specific notes.

---

## 5) Privacy & Licensing

- The raw HTML is redistributed via the Hugging Face dataset hub. Please respect the dataset provider’s license and terms of use.  
- This repository includes only **derived metadata** and code.  
- Do not commit large HTML files, graph tensors (`*.pt`), or model checkpoints to Git.

---

## 6) Troubleshooting

- **403/401 when downloading** → run `huggingface-cli login` or set `HUGGINGFACEHUB_API_TOKEN`.
- **Path not found** → double-check `--path-in-repo` (e.g., `web_ie_multidomain/app.zip`) and that it exists on the dataset hub.
- **Runtime errors about missing HTML** → confirm the local layout matches the tree above, or update your config paths.
- **Slow IO / too many files** → keep HTML zipped and ensure your pipeline streams from zip where possible.

If issues persist, open a GitHub Issue with your environment details and the exact command you ran.
