# 🚀 Quick Start Guide

## Option 1: Windows (Fastest)
1. Double-click `start.bat`
2. Wait for dependencies to install
3. Browser opens automatically at http://127.0.0.1:5000

## Option 2: Manual Setup (All Platforms)
```bash
# Install dependencies
pip install -r requirements.txt

# Run application  
python app.py
```

## Option 3: Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

## 📝 Testing the Application

1. **Upload the sample file**: Use `sample_input.xlsx` (10 test products)
2. **Enter API key**: Get from Mistral AI or Google AI
3. **Select model**: Mistral Small (faster) or Gemini 1.5 Flash (higher quality)
4. **Click "Start Generation"**: Watch real-time progress
5. **Download results**: Excel file with generated descriptions

## 🔑 Getting API Keys

### Mistral AI (Recommended for speed):
- Visit: https://console.mistral.ai/
- Create account → Get API key
- Model: mistral-small-latest

### Google Gemini (Recommended for quality):
- Visit: https://makersuite.google.com/
- Get API key for Gemini
- Model: gemini-1.5-flash

## 📊 Expected Output

For each product, you'll get:
- **Short Description**: 4 bullet points, no punctuation at end
- **Long Description**: 7-8 lines, SEO-optimized, professional

## 🛡️ Security Notes

- API keys are never stored or logged
- Files are automatically cleaned up after processing
- All data processing happens locally except AI API calls

## ⚡ Performance Tips

- **Small files (<100 products)**: 2-5 minutes
- **Medium files (100-1000)**: 10-30 minutes  
- **Large files (1000+)**: 1+ hours with periodic saves

## 🔧 Troubleshooting

- **Port 5000 busy**: Change port in app.py (line: `app.run(port=5000)`)
- **API errors**: Check internet connection and API key validity
- **File too large**: Split into smaller files (<5000 products each)

## 📁 File Structure
```
pharma-description-generator/
├── app.py              # Main Flask application
├── llm_client.py      # AI model integration  
├── utils.py           # Excel processing utilities
├── requirements.txt   # Dependencies
├── start.bat         # Windows startup script
├── start.sh          # Linux/Mac startup script
├── sample_input.xlsx # Test file with 10 products
├── templates/        # Web interface
├── static/          # CSS styling
└── README.md        # Full documentation
```

---
**Ready to generate professional pharmaceutical descriptions at scale!** 🧬
