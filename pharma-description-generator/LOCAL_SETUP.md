# Local Setup Instructions for Pharma Description Generator

These steps will help you set up and run the Pharma Description Generator on your local Windows device.

---

## 1. Prerequisites
- **Python 3.9+** (Recommended: 3.9, 3.10, or 3.11)
- **Git** (for cloning from GitHub)
- **Internet connection** (for installing dependencies and using LLM APIs)

## 2. Download the Project
- If you received a ZIP file, extract it to a folder (e.g., `C:\Users\yourname\Downloads\client4`).
- **Or, to clone from your GitHub repository:**
  ```powershell
  git clone <your-github-repo-url>
  cd pharma-description-generator
  ```
  Replace `<your-github-repo-url>` with your actual repository link.

## 3. Open a Terminal
- Open **Windows PowerShell** or **Command Prompt**.
- Navigate to the project folder:
  ```powershell
  cd path\to\pharma-description-generator
  ```

## 4. Create a Virtual Environment (Recommended)
```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 5. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Set API Keys
- You will need API keys for OpenRouter (Mistral 7B) and/or Gemini.
- You can provide these in the web UI when prompted, or set them as environment variables:
  ```powershell
  $env:OPENROUTER_API_KEY = "your-openrouter-key"
  $env:GEMINI_API_KEY = "your-gemini-key"
  ```

## 7. Run the Application
```powershell
python app.py
```
- The app will start and show a local address (e.g., `http://127.0.0.1:5000`).
- Open this address in your web browser.

## 8. Usage
- Upload your Excel file with product data (see template for required columns).
- Enter your API key(s) and select the model.
- Click **Start** to generate descriptions.
- Download the output Excel file when processing is complete.

## 9. Troubleshooting
- If you see errors about missing packages, re-run `pip install -r requirements.txt`.
- For API errors, check your API key and internet connection.
- For other issues, check the terminal for error messages.

## 10. Stopping the App
- Press `Ctrl+C` in the terminal to stop the server.

---

**For further help, contact your project provider.**
