# BuddyAI

BuddyAI is a Flask-based chatbot with intent matching, file upload support, PDF resume parsing, and CV-based interview question generation.

## Local Run

```powershell
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:HF_TOKEN="hf_your_token_here"
$env:HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
python app.py
```

## Render Deploy

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the GitHub repo.
3. Render will detect `render.yaml` automatically.
4. In Render dashboard, set `HF_TOKEN` as a secret environment variable.
5. Deploy.

Render uses:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

## Notes

- Uploaded files are kept in process memory only; they are not persisted across restarts.
- PDF uploads require `pdfplumber`, already listed in `requirements.txt`.
- Rotate any tokens that were previously shared in chat.