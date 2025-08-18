
# 🧬 Pharmaceutical Description Generator

A production-ready Python web application for generating standardized product descriptions for pharmaceutical and healthcare products using AI models (Mistral 7B via OpenRouter & Gemini 1.5 Flash).


## ✨ Features

- **AI-Powered Generation**: Uses Mistral 7B (OpenRouter) or Gemini 1.5 Flash for high-quality, compliant descriptions
- **Batch Processing**: Efficiently handles up to 50,000 products with parallel/concurrent processing
- **Two Description Types**:
    - **Short**: 4 concise bullet points (no punctuation at end)
    - **Long**: 7-8 lines, SEO-optimized, professional content (no title, no bullet points)
- **Web Interface**: Clean, modern Bootstrap UI (single-page)
- **Real-time Progress**: Live progress tracking, estimated time, and status
- **Excel Integration**: Direct Excel input/output with professional formatting
- **Secure**: API keys never stored, used only during processing session
- **Automatic Cleanup**: Uploads and outputs are auto-removed after 24h
- **No Test or Sample Files**: Only production code is included for client delivery

## 📋 Requirements

- Python 3.8 or higher
- Internet connection for AI API calls
- API key for Mistral AI or Google Gemini

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open in Browser

Navigate to: **http://127.0.0.1:5000**


## 📁 Project Structure

```
pharma-description-generator/
│
├── app.py                 # Main Flask application (production only)
├── llm_client.py          # LLM integration (Mistral & Gemini)
├── utils.py               # Excel handling and utilities
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── templates/
│   └── index.html         # Web interface (Bootstrap + JS)
│
├── static/
│   └── style.css          # Custom styling
│
├── uploads/               # Temporary upload storage (auto-created, auto-cleaned)
└── output/                # Generated output files (auto-created, auto-cleaned)
```

## 📊 Input Format

Your Excel file should have:
- **Column A**: Product/medicine names (required)
- **Additional columns**: Ignored during processing
- **File format**: .xlsx or .xls
- **Maximum size**: 50MB
- **Maximum products**: 50,000

### Example Input:
```
Product Name
Aspirin 100mg Tablets
Ibuprofen 200mg Capsules
Paracetamol 500mg Tablets
Vitamin D3 1000 IU Softgels
```

## 📈 Output Format

Generated Excel file contains:
- **Column A**: Product Name (original)
- **Column B**: Short Description (4 bullet points)
- **Column C**: Long Description (7-8 lines, SEO-optimized)

### Example Output:
```
Product Name          | Short Description           | Long Description
Aspirin 100mg Tablets | • Pain relief medication   | Aspirin 100mg tablets provide effective...
                      | • Anti-inflammatory        | These pharmaceutical-grade tablets are...
                      | • Cardiovascular support   | Formulated for optimal absorption and...
                      | • Doctor recommended       | Suitable for daily use as directed by...
```

## 🔑 API Keys

### For Mistral AI:
1. Visit: https://console.mistral.ai/
2. Create account and get API key
3. Select "Mistral Small" in the application

### For Google Gemini:
1. Visit: https://makersuite.google.com/
2. Get API key for Gemini
3. Select "Gemini 1.5 Flash" in the application

**Security Note**: API keys are only used during processing and never stored.

## 🎯 Usage Instructions

### Step 1: Upload File
- Click "Choose File" and select your Excel file
- File will be validated automatically

### Step 2: Configure
- Enter your API key (Mistral or Gemini)
- Select AI model (Mistral for speed, Gemini for quality)

### Step 3: Process
- Click "Start Generation"
- Monitor real-time progress
- Processing runs in background

### Step 4: Download
- Download button appears when complete
- File includes all generated descriptions

## ⚡ Performance

| Products | Mistral Small | Gemini 1.5 Flash |
|----------|---------------|-------------------|
| 100      | ~2 minutes    | ~3 minutes        |
| 1,000    | ~15 minutes   | ~20 minutes       |
| 5,000    | ~1 hour       | ~1.5 hours        |
| 10,000   | ~2 hours      | ~3 hours          |

*Times are estimates and may vary based on API response times*

## 🛠 Advanced Configuration

### Batch Size
Default: 5 products per batch (configurable in `llm_client.py`)

```python
processor = BatchProcessor(llm_client, batch_size=5)
```

### Retry Settings
Default: 3 retries with exponential backoff

```python
await self._call_mistral(prompt, max_retries=3)
```

### File Size Limits
Large files (>5,000 products) are automatically split into multiple output files.

## 🔧 Troubleshooting

### Common Issues:

**"API key invalid"**
- Verify API key is correct and active
- Check if you have sufficient API credits

**"File too large"**
- Maximum file size is 50MB
- Split large files into smaller batches

**"Processing failed"**
- Check internet connection
- Verify API service is operational
- Try with a smaller batch first

**"Empty descriptions generated"**
- May indicate API rate limiting
- Try again after a few minutes
- Consider switching to different model

### Error Recovery:
- Partial results are saved every 100 products
- Failed jobs can be restarted from beginning
- Check console logs for detailed error information

## 📝 Customization

### Modify Prompts
Edit prompts in `llm_client.py`:

```python
def _get_prompt(self, product_name: str, description_type: str) -> str:
    if description_type == 'short':
        return f"Your custom short prompt for {product_name}..."
    elif description_type == 'long':
        return f"Your custom long prompt for {product_name}..."
```

### Adjust Batch Size
Modify in `app.py`:

```python
processor = BatchProcessor(llm_client, batch_size=10)  # Increase for faster processing
```

### Change Output Format
Modify Excel formatting in `utils.py`:

```python
def _format_worksheet(worksheet):
    # Add your custom formatting here
```

## 🔒 Security & Privacy

- **API Keys**: Never logged or stored permanently
- **File Data**: Uploaded files deleted after processing
- **Output Files**: Automatically cleaned up after 24 hours
- **Network**: All API calls use HTTPS encryption
- **Local Processing**: No data sent to third parties except AI APIs

## 🌟 Production Deployment

### Using Gunicorn (Recommended):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

### Environment Variables:
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```


## 🧹 Workspace Cleanup

The application automatically manages all temporary files for you:

- **Output files**: Automatically removed after 24 hours
- **Upload files**: Cleaned when processing completes
- **Temporary data**: Cleared on application restart

No test, sample, or demo files are present in this workspace. Only production code and assets are included for client delivery.

**Note**: The workspace includes a `.gitignore` file to prevent output files from being tracked in version control.


## 📞 Support

### Self-Help:
1. Check console output for error details
2. Verify API key and credits
3. Test with a small file first
4. Review input file format

### Technical Details:
- **Framework**: Flask 2.x
- **AI Libraries**: Google GenerativeAI, httpx
- **Excel**: Pandas + OpenPyXL
- **Frontend**: Bootstrap 5.3 + JavaScript
- **Async**: AsyncIO for concurrent processing

## 📄 License

This software is provided as-is for pharmaceutical and healthcare businesses. Modify and extend as needed for your specific requirements.

## 🔄 Version History

- **v1.0** - Initial release with Mistral & Gemini support
- **v1.1** - Added progress tracking and large file handling  
- **v1.2** - Enhanced error recovery and batch processing

---

**Ready to generate professional pharmaceutical descriptions at scale!** 🚀

For additional customization or enterprise features, the codebase is fully documented and modular for easy extension.
