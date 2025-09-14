# Streamlit Cloud Deployment

## Quick Start

This app is now **Python 3.13 compatible** and should deploy successfully on Streamlit Cloud.

### Requirements File
- **Main**: `requirements.txt` (Python 3.13 compatible, no PaddleOCR)
- **Alternative**: `requirements-py313.txt` (same content)

### OCR Engines Available
- ✅ **Tesseract** - Works on all Python versions
- ✅ **TrOCR** - Transformer-based OCR
- ✅ **Donut** - Layout-aware Vision Transformer
- ❌ **PaddleOCR** - Not available on Python 3.13

### Environment Variables
Set these in Streamlit Cloud:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Deployment Steps
1. Connect your GitHub repository to Streamlit Cloud
2. Streamlit Cloud will automatically detect `requirements.txt`
3. The app will install successfully without PaddleOCR conflicts
4. Set your OpenAI API key in the environment variables
5. Deploy!

### Features
- 📄 PDF and image document processing
- 🔍 Multiple OCR engines (Tesseract, TrOCR, Donut)
- 🤖 AI-powered text normalization
- 📊 Structured JSON extraction
- 📈 XLSX export with preview
- 🌍 Russian language support

The app will automatically fall back to available OCR engines if PaddleOCR is not installed.
