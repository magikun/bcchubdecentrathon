# Intelligent OCR for Banking Documents

MVP system that combines modern OCR (PaddleOCR, TrOCR, Donut) with LLM-based post-processing to extract structured JSON from banking documents (checks, contracts, statements). Includes Streamlit demo and evaluation vs. Tesseract.

## Features

- Robust OCR: PaddleOCR, TrOCR, Donut; baseline Tesseract
- Preprocessing: denoise, thresholding, upscaling
- LLM post-processing to strict JSON schema (Pydantic)
- Metrics: CER, WER, normalized Levenshtein; field-level Accuracy/Precision/Recall/F1; exact match
- Noisy subset evaluation via noise score
- Streamlit demo with uploads, JSON/text export

## Setup

### 1. Install Dependencies
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure LLM (Optional but Recommended)
For structured JSON extraction, you need an OpenAI API key:

1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"  # Optional: use faster/cheaper model
```
3. Or set environment variable: `export OPENAI_API_KEY="sk-..."`

**Without API key**: App works but only extracts raw OCR text (no structured JSON).

### Windows Quickstart (persistent venv)

You can use the included scripts to ensure dependencies persist across IDE restarts:

```powershell
# One-time or after pulling changes
./scripts/setup.ps1           # creates .venv and installs requirements

# Run the app (auto-bootstraps if .venv missing)
./scripts/run_app.ps1
```

The project also includes `.vscode/settings.json` which pins the interpreter to `.venv\\Scripts\\python.exe` so VS Code uses the same environment every time.

Optionally set LLM:

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Run Demo

```bash
streamlit run app.py
```

### Batch processing

- In the app sidebar, enable "Batch processing (folder or multi-file)".
- Either upload multiple files at once or provide a folder path containing PDFs/images.
- Download the run summary JSONL after completion.

## Evaluation

Prepare a dataset directory with images and optional per-image ground truth JSONs named `<stem>.json` following `src/schema/bank_docs.py`.

```bash
python -m src.eval.run_eval --images path/to/images --gts path/to/gts --engine paddleocr --compare_tesseract --out eval_results.json
python -m src.eval.aggregate --results eval_results.json
```

The script writes per-file metrics including baseline Tesseract for comparison and aggregates improvements in percent, including a noisy subset (noise_score >= 0.5).

## Notes

- Donut/LayoutLMv3-like models are heavier; ensure GPU if possible.
- PaddleOCR may require specific runtime packages; see Paddle docs for Windows.
- This is an MVP; fine-tuning and domain prompts can further improve extraction.
