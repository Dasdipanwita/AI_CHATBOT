// Function to format time
function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ─── Voice Input (Web Speech API) ───────────────────────────────────────────
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isRecording = false;

if (SpeechRecognitionAPI) {
    recognition = new SpeechRecognitionAPI();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById('user-input').value = transcript;
        stopRecording();
        sendMessage(null);
    };
    recognition.onerror = () => stopRecording();
    recognition.onend   = () => stopRecording();
}

function startRecording() {
    if (!recognition) { addMessage('Voice input is not supported in this browser.'); return; }
    isRecording = true;
    recognition.start();
    const btn = document.getElementById('mic-btn');
    if (btn) btn.classList.add('recording');
}

function stopRecording() {
    isRecording = false;
    const btn = document.getElementById('mic-btn');
    if (btn) btn.classList.remove('recording');
    try { recognition && recognition.stop(); } catch(_) {}
}

// ─── Voice Output (Web Speech Synthesis) ────────────────────────────────────
let voiceEnabled = false;

function toggleVoice() {
    voiceEnabled = !voiceEnabled;
    const btn = document.getElementById('voice-toggle');
    if (btn) {
        btn.classList.toggle('active', voiceEnabled);
        btn.title = voiceEnabled ? 'Voice output ON' : 'Enable voice output';
        btn.querySelector('i').className = voiceEnabled
            ? 'fa-solid fa-volume-high'
            : 'fa-solid fa-volume-xmark';
    }
}

function speak(text) {
    if (!voiceEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    // Strip markdown syntax so it sounds natural
    const plain = text
        .replace(/```[\s\S]*?```/g, 'code block')
        .replace(/[#*`_>]/g, '')
        .slice(0, 500);
    const utt = new SpeechSynthesisUtterance(plain);
    utt.lang = 'en-US';
    utt.rate = 1.0;
    window.speechSynthesis.speak(utt);
}

// ─── Chat History (SQLite via /history + /clear_chat) ───────────────────────
async function loadHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();
        const container = document.getElementById('history-list');
        if (!container) return;
        container.innerHTML = '';
        if (!data.history || data.history.length === 0) {
            container.innerHTML = '<p class="no-history">No history yet.</p>';
            return;
        }
        data.history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.textContent = item.user_message.length > 38
                ? item.user_message.slice(0, 38) + '…'
                : item.user_message;
            div.title = item.user_message;
            // Click to prefill the input box
            div.addEventListener('click', () => {
                document.getElementById('user-input').value = item.user_message;
                document.getElementById('user-input').focus();
            });
            container.appendChild(div);
        });
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

async function clearChat() {
    try { await fetch('/clear_chat', { method: 'POST' }); } catch(_) {}
    const msgs = document.getElementById('chat-messages');
    msgs.innerHTML = '';
    addMessage('Hello! I am BuddyAI. How can I assist you today?');
    const container = document.getElementById('history-list');
    if (container) container.innerHTML = '<p class="no-history">Chat cleared.</p>';
}

// ─── DOMContentLoaded ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const attachButton = document.getElementById('attach-btn');
    const fileInput = document.getElementById('file-input');

    if (attachButton && fileInput) {
        attachButton.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileUpload);
    }

    const micBtn = document.getElementById('mic-btn');
    if (micBtn) {
        micBtn.addEventListener('click', () => {
            isRecording ? stopRecording() : startRecording();
        });
    }

    const chatForm = document.getElementById('chat-form');
    if (chatForm) {
        chatForm.addEventListener('submit', function(event) {
            event.preventDefault();
            sendMessage(event);
        });
    }

    // Load history into sidebar on page load
    loadHistory();
});


// Function to add a message to the chat
function addMessage(text, isUser = false) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textP = document.createElement('div');
    if (!isUser && typeof marked !== 'undefined') {
        textP.innerHTML = marked.parse(text);
    } else {
        textP.textContent = text;
    }
    
    const timeSpan = document.createElement('span');
    timeSpan.className = 'time';
    timeSpan.textContent = getCurrentTime();
    
    contentDiv.appendChild(textP);
    contentDiv.appendChild(timeSpan);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Function to scroll chat to bottom
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to show/hide typing indicator
function toggleTypingIndicator(show) {
    const indicator = document.getElementById('typing-indicator');
    indicator.style.display = show ? 'flex' : 'none';
    if(show) scrollToBottom();
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    addMessage(`Uploaded file: ${file.name}`, true);
    toggleTypingIndicator(true);

    try {
        const response = await fetch('/upload_file', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        toggleTypingIndicator(false);

        if (!response.ok) {
            addMessage(data.response || 'Unable to upload that file right now.');
        } else {
            addMessage(data.response || `Uploaded ${file.name}. Ask a question about it.`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        toggleTypingIndicator(false);
        addMessage('Sorry, I could not upload that file.');
    } finally {
        event.target.value = '';
    }
}

// Handle sending message
async function sendMessage(event) {
    if (event) event.preventDefault(); // Prevent form submission
    
    const inputField = document.getElementById('user-input');
    const message = inputField.value.trim();
    
    if (!message) return;
    
    // Add user message to UI
    addMessage(message, true);
    inputField.value = ''; // clear input
    
    // Show typing indicator
    toggleTypingIndicator(true);
    
    try {
        // Send request to Flask backend
        const response = await fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Hide typing indicator and add bot response
        setTimeout(() => {
            toggleTypingIndicator(false);
            addMessage(data.response);
            speak(data.response);      // read aloud if voice is enabled
            loadHistory();             // refresh sidebar history
        }, 500 + Math.random() * 500); // Add a small synthetic delay for realism
        
    } catch (error) {
        console.error('Error:', error);
        toggleTypingIndicator(false);
        addMessage("Sorry, I encountered an error connecting to the server.", false);
    }
}
