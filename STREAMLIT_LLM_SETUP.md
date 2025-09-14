# LLM Setup for Streamlit Cloud Deployment

## Quick Setup

### 1. Get OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to [API Keys](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Copy the key (starts with `sk-...`)

### 2. Deploy to Streamlit Cloud
1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Click "New app"
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Click "Advanced settings"

### 3. Add Secrets
In the "Secrets" section, add:

```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"
```

**Important**: Replace `sk-your-actual-api-key-here` with your real API key!

### 4. Deploy
Click "Deploy!" and wait for the app to build.

## Environment Variables

The app reads these environment variables:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | ✅ Yes |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` | ❌ No |
| `OPENAI_ORG` | Organization ID | - | ❌ No |
| `OPENAI_BASE_URL` | Custom base URL | - | ❌ No |

## Testing LLM Functionality

### 1. Check API Key
The app will show:
- ✅ "LLM Processing: Available" if API key is set
- ❌ "LLM Processing: Not available" if no API key

### 2. Test with Document
1. Upload a PDF or image
2. Select an OCR engine
3. Enable "AI Normalization" if needed
4. Click "Process Document"
5. Check the JSON output for structured data

### 3. Expected JSON Structure
```json
{
  "schema_version": "1.0",
  "ocr_engine": "PaddleOCR",
  "ocr_text": "Clean extracted text...",
  "contract": {
    "doc_type": "contract",
    "parties": {
      "seller": {
        "name": "Company Name",
        "inn": "1234567890",
        "kpp": "123456789"
      },
      "buyer": {
        "name": "Another Company",
        "inn": "0987654321"
      }
    },
    "date": "2022-12-17",
    "contract_number": "SM-1712/22",
    "total_amount": 3209315.71,
    "currency": "RUB"
  }
}
```

## Troubleshooting

### LLM Not Working
1. **Check API Key**: Make sure it's correctly set in Streamlit secrets
2. **Check Credits**: Ensure you have OpenAI credits
3. **Check Model**: Try `gpt-4o-mini` (cheaper) instead of `gpt-4o`
4. **Check Logs**: Look at Streamlit Cloud logs for errors

### Common Errors
- `401 Unauthorized`: Invalid API key
- `429 Rate Limited`: Too many requests
- `Insufficient credits`: Need to add credits to OpenAI account

### Fallback Behavior
If LLM fails, the app will:
1. Use basic text normalization
2. Still extract OCR text
3. Show warning message
4. Continue working without structured JSON

## Cost Optimization

### Use Fast Mode
- Enable "Fast Mode" in the app
- Uses `gpt-4o-mini` (cheaper model)
- Faster processing
- Good quality for most documents

### Model Comparison
| Model | Cost | Speed | Quality |
|-------|------|-------|---------|
| `gpt-4o-mini` | $0.15/1M tokens | Fast | Good |
| `gpt-4o` | $2.50/1M tokens | Slow | Excellent |

## Security Notes

- ✅ API key is stored securely in Streamlit secrets
- ✅ Never commit API keys to Git
- ✅ Use environment variables only
- ✅ Rotate keys regularly

## Support

If you need help:
1. Check Streamlit Cloud logs
2. Test API key at [OpenAI Playground](https://platform.openai.com/playground)
3. Verify credits in [OpenAI Usage](https://platform.openai.com/usage)
