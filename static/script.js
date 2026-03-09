// Function to format time
function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

document.addEventListener('DOMContentLoaded', () => {
    const attachButton = document.getElementById('attach-btn');
    const fileInput = document.getElementById('file-input');

    if (attachButton && fileInput) {
        attachButton.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileUpload);
    }
});

// Function to add a message to the chat
function addMessage(text, isUser = false) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textP = document.createElement('p');
    textP.textContent = text;
    
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
    event.preventDefault(); // Prevent form submission
    
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
        }, 500 + Math.random() * 500); // Add a small synthetic delay for realism
        
    } catch (error) {
        console.error('Error:', error);
        toggleTypingIndicator(false);
        addMessage("Sorry, I encountered an error connecting to the server.", false);
    }
}
