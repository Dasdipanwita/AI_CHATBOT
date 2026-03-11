from flask import Flask, render_template, request, jsonify, session
from chatbot import ChatBot
import database
import nltk
import os
import io
import secrets

try:
    import pdfplumber
except Exception:
    pdfplumber = None

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(16))

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

# Initialize the SQLite chat history database
database.init_db()

SUPPORTED_UPLOAD_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.css', '.xml', '.yml', '.yaml', '.pdf'}
MAX_UPLOAD_BYTES = 1_000_000
CONTENT_CHAR_LIMIT = 15000


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

    # Ensure every visitor has a persistent session_id for DB storage
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)

    # Retrieve per-session conversation history for context memory
    chat_history = session.get('chat_history', [])

    try:
        context_content = session.get('content')
        context_filename = session.get('filename')
        if context_content:
            response = bot.get_response(
                user_message,
                context_text=context_content,
                context_name=context_filename,
                chat_history=chat_history,
            )
        else:
            response = bot.get_response(user_message, chat_history=chat_history)

        # Update in-session memory (keep last 4 exchanges = 8 messages)
        chat_history.append({'role': 'user', 'content': user_message})
        chat_history.append({'role': 'assistant', 'content': response})
        session['chat_history'] = chat_history[-8:]

        # Persist to SQLite
        database.save_message(session['session_id'], user_message, response)

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

    session['filename'] = filename
    truncated = content[:CONTENT_CHAR_LIMIT]
    session['content'] = truncated
    note = ' Note: the file is long — I can only read the first portion of it.' if len(content) > CONTENT_CHAR_LIMIT else ''
    return jsonify({'response': f'Uploaded {filename}. I can now answer questions about this file.{note}'})

@app.route('/history', methods=['GET'])
def get_history():
    """Return the chat history for the current session (latest 20 exchanges)."""
    session_id = session.get('session_id', '')
    if not session_id:
        return jsonify({'history': []})
    return jsonify({'history': database.get_history(session_id, limit=20)})


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Delete the chat history for the current session and reset in-memory context."""
    session_id = session.get('session_id', '')
    if session_id:
        database.clear_history(session_id)
    session.pop('chat_history', None)
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    # Creating templates and static folders if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False,
        use_reloader=False,
    )
