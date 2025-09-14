# Deployment Guide

## Streamlit Cloud Deployment

### For Python 3.13+ Environments

If you're deploying to Streamlit Cloud with Python 3.13, use the Python 3.13 compatible requirements:

```bash
# Use this requirements file for Python 3.13
requirements-py313.txt
```

### For Python 3.11-3.12 Environments

Use the standard requirements file:

```bash
# Use this requirements file for Python 3.11-3.12
requirements.txt
```

## Manual Installation

### Option 1: Smart Installer (Recommended)
```bash
python install_requirements.py
```

### Option 2: Direct pip install
```bash
# For Python 3.13+
pip install -r requirements-py313.txt

# For Python 3.11-3.12
pip install -r requirements.txt
```

### Option 3: Minimal installation
```bash
pip install -r requirements-minimal.txt
```

## OCR Engine Compatibility

| Engine | Python 3.11 | Python 3.12 | Python 3.13 |
|--------|-------------|-------------|-------------|
| Tesseract | ✅ | ✅ | ✅ |
| TrOCR | ✅ | ✅ | ✅ |
| Donut | ✅ | ✅ | ✅ |
| PaddleOCR | ✅ | ✅ | ❌ |

## Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

## Troubleshooting

### PaddlePaddle Installation Issues
- **Python 3.13**: PaddlePaddle is not compatible. Use Tesseract or TrOCR instead.
- **Windows**: May require Visual Studio Build Tools
- **Linux**: Usually works without issues

### Streamlit Cloud Issues
- Use `requirements-py313.txt` for Python 3.13
- Ensure all dependencies are listed
- Check the deployment logs for specific errors

## Running the App

```bash
streamlit run app.py
```

Or use the PowerShell script:
```powershell
.\scripts\run_app.ps1
```
