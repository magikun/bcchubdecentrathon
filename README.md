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

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

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
