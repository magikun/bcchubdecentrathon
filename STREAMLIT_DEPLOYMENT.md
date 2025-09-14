# Streamlit Cloud Deployment Guide

## Quick Deploy

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Click "New app"**
4. **Select your forked repository**
5. **Set the main file path**: `app.py`
6. **Add your OpenAI API key** in the secrets section:
   ```
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```
7. **Click "Deploy!"**

## Configuration Files Added

### `.streamlit/config.toml`
- Optimized server settings for cloud deployment
- Disabled usage stats collection
- Set proper theme colors

### `packages.txt`
- System dependencies: `tesseract-ocr` and `poppler-utils`
- Required for OCR and PDF processing

### `requirements.txt`
- **Python 3.13 compatible** - removed PaddleOCR/PaddlePaddle
- **OpenCV included** - for image processing
- **All ML frameworks** - PyTorch, Transformers, ONNX Runtime
- **PDF processing** - pdf2image, PyPDF2

## OCR Engines Available

- ✅ **Tesseract** - Works on all Python versions
- ✅ **TrOCR** - Transformer-based OCR
- ✅ **Donut** - Layout-aware Vision Transformer
- ❌ **PaddleOCR** - Excluded for Python 3.13 compatibility

## Environment Variables

Set these in Streamlit Cloud secrets:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"  # Optional
OPENAI_ORG_ID = "org-..."     # Optional
```

## Troubleshooting

### If deployment fails:
1. Check that all dependencies are compatible
2. Ensure OpenAI API key is set correctly
3. Verify the main file path is `app.py`

### If OCR doesn't work:
1. Tesseract should work automatically
2. TrOCR and Donut require internet for model downloads
3. Check the app logs for specific errors

## Features

- 📄 **PDF Processing** - Upload PDFs and extract text
- 🔍 **Multiple OCR Engines** - Tesseract, TrOCR, Donut
- 🤖 **AI Post-processing** - OpenAI GPT for structured extraction
- 📊 **XLSX Export** - Download results as Excel files
- 🌍 **Russian Language** - Optimized for Cyrillic text
- 🎨 **Modern UI** - Clean Streamlit interface

## Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify your OpenAI API key is valid
3. Ensure you have sufficient OpenAI credits
