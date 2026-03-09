from flask import Flask, render_template, request, jsonify
from chatbot import ChatBot
import nltk
import os
import io

try:
    import pdfplumber
except Exception:
    pdfplumber = None

app = Flask(__name__)

# Download necessary NLTK corpora upon startup
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK datasets: {e}")

# Initialize the ChatBot instance
# This will load intents, initialize vectorizer and pre-calculate tfidf
bot = ChatBot('intents.json')

SUPPORTED_UPLOAD_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.css', '.xml', '.yml', '.yaml', '.pdf'}
MAX_UPLOAD_BYTES = 1_000_000
uploaded_context = {
    'filename': None,
    'content': None,
}


def extract_upload_text(raw_bytes, extension):
    """Extract readable text from supported upload types."""
    if extension == '.pdf':
        if pdfplumber is None:
            raise ValueError('PDF support is not available on the server right now.')

        extracted_pages = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ''
                if page_text.strip():
                    extracted_pages.append(page_text.strip())

        return '\n\n'.join(extracted_pages).strip()

    return raw_bytes.decode('utf-8', errors='ignore').strip()

@app.route('/')
def home():
    """Renders the main chat interface."""
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    """Handles messages sent from the frontend and returns a bot response."""
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'response': "Error: Empty message received"}), 400
        
    try:
        if uploaded_context['content']:
            response = bot.get_response(
                user_message,
                context_text=uploaded_context['content'],
                context_name=uploaded_context['filename'],
            )
        else:
            response = bot.get_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500


@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Upload a supported text file and make its contents available for follow-up questions."""
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({'response': 'Please choose a file to upload.'}), 400

    filename = os.path.basename(file.filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ', '.join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        return jsonify({'response': f'Unsupported file type. Please upload one of: {allowed}'}), 400

    raw = file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({'response': 'File is too large. Please upload a file smaller than 1 MB.'}), 400

    try:
        content = extract_upload_text(raw, extension)
    except Exception as exc:
        return jsonify({'response': f'Could not read that file: {exc}'}), 400

    if not content:
        return jsonify({'response': 'That file appears to be empty or unreadable.'}), 400

    uploaded_context['filename'] = filename
    uploaded_context['content'] = content[:15000]
    return jsonify({'response': f'Uploaded {filename}. I can now answer questions about this file.'})

if __name__ == '__main__':
    # Creating templates and static folders if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    # Disable the reloader and debug mode when running inside environments
    # that don't run the script in the main interpreter thread (e.g. Streamlit).
    # This prevents `signal only works in main thread` errors coming from
    # Werkzeug's reloader which registers signal handlers.
    app.run(debug=False, port=5000, use_reloader=False)
