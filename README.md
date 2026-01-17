  Pharmaceutical Description Generator

A **production-ready Python web application** for generating **standardized short and long product descriptions** for pharmaceutical and healthcare products using AI models (Mistral 7B via OpenRouter & Gemini 1.5 Flash).

---

##  Features

* **AI-Powered Generation** → Leverages Mistral 7B (OpenRouter) or Gemini 1.5 Flash for compliant, professional descriptions
* **Batch Processing** → Handles up to **50,000 products** with concurrent execution
* **Two Description Types**:

  * **Short:** 4 concise bullet points (no punctuation at end)
  * **Long:** 7–8 SEO-optimized lines (no title, no bullets)
* **Excel Integration** → Input & output directly in Excel with professional formatting
* **Web Interface** → Clean, Bootstrap-based single-page UI
* **Real-time Progress** → Live status updates with time estimation
* **Secure by Design** → API keys used only in session, never stored
* **Automatic Cleanup** → Uploads & outputs auto-deleted after 24h
* **Production-Ready** → No test/demo files, only client-grade code

---

##  Requirements

* Python **3.8+**
* Internet connection for API calls
* API key for **Mistral AI** or **Google Gemini**

---

##  Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**

   ```bash
   python app.py
   ```

3. **Open in browser**
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

##  Project Structure

```
pharma-description-generator/
├── app.py            # Flask application (production)
├── llm_client.py     # Mistral & Gemini integration
├── utils.py          # Excel handling utilities
├── requirements.txt  # Dependencies
├── templates/
│   └── index.html    # Web interface (Bootstrap)
├── static/
│   └── style.css     # Styling
├── uploads/          # Temporary input files (auto-cleaned)
└── output/           # Generated Excel files (auto-cleaned)
```

---

##  Input Format

* **Column A:** Product/medicine names (required)
* Other columns → ignored
* Format: `.xlsx` or `.xls`
* Max file size: **50MB**
* Max records: **50,000**

**Example Input:**

```
Product Name
Aspirin 100mg Tablets
Ibuprofen 200mg Capsules
Paracetamol 500mg Tablets
Vitamin D3 1000 IU Softgels
```

---

##  Output Format

* **Column A:** Product Name (original)
* **Column B:** Short Description (4 bullet points)
* **Column C:** Long Description (7–8 SEO-optimized lines)

---

## API Keys

**Mistral AI:**

*https://openrouter.ai/mistralai/mistral-7b-instruct:free → Generate key
* Select **Mistral7B**

**Google Gemini:**

* [https://makersuite.google.com](https://makersuite.google.com) → Get key
* Select **Gemini 1.5 Flash**

 Keys are **never stored**, only used for the current process.

---

##  Usage Instructions

1. **Upload File** → Select Excel file (validated automatically)
2. **Configure** → Enter API key & choose model (Mistral = speed, Gemini = quality)
3. **Process** → Click *Start Generation* & track live progress
4. **Download** → Output file ready with added descriptions

---

##  Performance (Estimates)

| Products | Mistral Small | Gemini 1.5 Flash |
| -------- | ------------- | ---------------- |
| 100      | \~2 min       | \~3 min          |
| 1,000    | \~15 min      | \~20 min         |
| 5,000    | \~1 hr        | \~1.5 hr         |
| 10,000   | \~2 hr        | \~3 hr           |

---

##  Advanced Configuration

* **Batch Size (default 5)** → configurable in `llm_client.py`
* **Retries (default 3)** → exponential backoff
* **Large Files** (>5k rows) → automatically split into smaller output files

---

##  Troubleshooting

* **Invalid API Key** → Check key & credits
* **File too large** → Max 50MB
* **Processing failed** → Retry with smaller batch, check connection
* **Empty outputs** → Could be API rate limits; retry after delay

 Partial results saved every 100 products for recovery

---

##  Customization

* **Modify Prompts** → In `llm_client.py`
* **Adjust Batch Size** → In `app.py`
* **Change Excel Formatting** → In `utils.py`

---

## Security & Privacy

* API keys → never logged or stored
* Uploaded files → deleted after processing
* Outputs → auto-cleaned in 24h
* Network → HTTPS encryption
* Only AI APIs receive data

---

##  Deployment

**Gunicorn** (recommended):

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Environment:**

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

---

##  Cleanup

* Outputs → auto-deleted in 24h
* Uploads → removed post-processing
* Temporary data → cleared on restart
* `.gitignore` excludes outputs from version control

---

##  Support

* Check console logs
* Verify API credits & connection
* Test with small file
* Review input format

**Tech stack:**

* Flask 2.x
* Google GenerativeAI, httpx
* Pandas + OpenPyXL
* Bootstrap 5.3 + JS
* AsyncIO for concurrency

---

##  License

Provided as-is for **pharmaceutical & healthcare businesses**. Extend/customize as needed.

---

##  Version History

* **v1.0** → Initial release with Mistral & Gemini support
* **v1.1** → Added progress tracking & large file handling
* **v1.2** → Improved error recovery & batch processing

---
 Ready to generate **professional pharma descriptions at scale!**


