// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentUser = null;
let authToken = localStorage.getItem('authToken');
let chatMessages = [];
let isProcessing = false;



// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
        ${message}
    `;
    
    const container = document.getElementById('notifications');
    container.appendChild(notification);
    
    // Show notification
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Hide and remove after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => container.removeChild(notification), 300);
    }, 5000);
}

// Show loading state
function showLoading(show = true) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

// Show/hide sections
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show requested section
    const targetSection = document.getElementById(sectionName + 'Section');
    if (targetSection) {
        targetSection.style.display = 'block';
    }
    
    // Special handling for chat
    if (sectionName === 'chat') {
        document.querySelector('.chat-container').style.display = 'block';
        scrollChatToBottom();
    }
    
    // Update URL hash
    window.location.hash = sectionName;
}

// API request helper
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    // Add auth token if available
    if (authToken) {
        defaultOptions.headers['Authorization'] = `Bearer ${authToken}`;
    }
    
    const requestOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };
    
    try {
        const response = await fetch(url, requestOptions);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || `HTTP error! status: ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error('API Request Error:', error);
        throw error;
    }
}

// =============================================================================
// AUTHENTICATION FUNCTIONS
// =============================================================================

// Initialize auth state
function initAuth() {
    if (authToken) {
        // Verify token and get user info
        getCurrentUser();
    } else {
        updateAuthUI(false);
    }
}

// Get current user info
async function getCurrentUser() {
    try {
        const response = await apiRequest('/api/auth/me');
        if (response.success !== false) {
            currentUser = response;
            updateAuthUI(true);
        }
    } catch (error) {
        console.error('Failed to get user info:', error);
        logout();
    }
}

// Update auth UI
function updateAuthUI(isLoggedIn) {
    const authButtons = document.getElementById('authButtons');
    const userMenu = document.getElementById('userMenu');
    const userName = document.getElementById('userName');
    
    if (isLoggedIn && currentUser) {
        authButtons.style.display = 'none';
        userMenu.style.display = 'flex';
        userName.textContent = `Welcome, ${currentUser.username || currentUser.first_name || 'User'}!`;
    } else {
        authButtons.style.display = 'flex';
        userMenu.style.display = 'none';
    }
}

// Login function
async function login(email, password) {
    try {
        showLoading(true);
        
        const response = await apiRequest('/api/auth/login', {
            method: 'POST',
            body: JSON.stringify({ email, password }),
        });
        
        if (response.access_token) {
            authToken = response.access_token;
            localStorage.setItem('authToken', authToken);
            
            await getCurrentUser();
            showNotification('Login successful! Welcome back.', 'success');
            showSection('home');
        } else {
            throw new Error('Invalid response format');
        }
    } catch (error) {
        console.error('Login error:', error);
        showNotification(error.message || 'Login failed. Please check your credentials.', 'error');
    } finally {
        showLoading(false);
    }
}




// Register function
async function register(userData) {
    try {
        showLoading(true);
        
        const response = await apiRequest('/api/auth/register', {
            method: 'POST',
            body: JSON.stringify(userData),
        });
        
        if (response.success) {
            showNotification('Account created successfully! Please login.', 'success');
            showSection('login');
        } else {
            throw new Error(response.message || 'Registration failed');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showNotification(error.message || 'Registration failed. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

// Logout function
function logout() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem('authToken');
    updateAuthUI(false);
    showNotification('Logged out successfully.', 'success');
    showSection('home');
}

// =============================================================================
// MENTAL HEALTH ASSESSMENT FUNCTIONS
// =============================================================================

// Process mental health assessment
async function processAssessment(textInput, assessmentType = 'general') {
    try {
        showLoading(true);
        
        const assessmentData = {
            text_input: textInput,
            assessment_type: assessmentType,
            previous_context: chatMessages.slice(-5).map(msg => msg.content) // Last 5 messages for context
        };
        
        const response = await apiRequest('/api/mental-health/assess', {
            method: 'POST',
            body: JSON.stringify(assessmentData),
        });
        
        if (response.success && response.data) {
            displayAssessmentResults(response.data);
            showSection('results');
        } else {
            throw new Error(response.message || 'Assessment failed');
        }
    } catch (error) {
        console.error('Assessment error:', error);
        showNotification(error.message || 'Assessment failed. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

// Display assessment results
function displayAssessmentResults(results) {
    const container = document.getElementById('resultsContent');
    
    // Risk level styling
    const riskClass = `risk-${results.risk_level}`;
    const riskIcon = {
        low: 'fa-smile',
        medium: 'fa-meh', 
        high: 'fa-frown',
        critical: 'fa-exclamation-triangle'
    }[results.risk_level] || 'fa-question';
    
    let html = `
        <div class="risk-level ${riskClass}">
            <h3><i class="fas ${riskIcon}"></i> Risk Level: ${results.risk_level.toUpperCase()}</h3>
            <p>Assessment completed with AI analysis</p>
        </div>
    `;
    
    // Crisis warning
    if (results.requires_immediate_attention) {
        html += `
            <div class="emergency-banner" style="margin-bottom: 1rem;">
                <strong>⚠️ IMMEDIATE ATTENTION NEEDED</strong><br>
                Please contact emergency services or crisis support immediately:
                <br><strong>988 - Crisis Lifeline | Text HOME to 741741</strong>
            </div>
        `;
    }
    
    // Predicted conditions
    if (results.predicted_conditions && results.predicted_conditions.length > 0) {
        html += `
            <div style="margin-bottom: 1rem;">
                <h4><i class="fas fa-brain"></i> AI Analysis Detected:</h4>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                    ${results.predicted_conditions.map(condition => 
                        `<span style="background: var(--primary); color: white; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.9rem;">
                            ${condition.replace('_', ' ').toUpperCase()}
                        </span>`
                    ).join('')}
                </div>
            </div>
        `;
    }
    
    // Crisis indicators
    if (results.crisis_indicators && results.crisis_indicators.length > 0) {
        html += `
            <div style="margin-bottom: 1rem; padding: 1rem; background: #fee2e2; border-radius: var(--radius);">
                <h4 style="color: #dc2626;"><i class="fas fa-warning"></i> Crisis Indicators Detected:</h4>
                <ul style="margin-top: 0.5rem; color: #dc2626;">
                    ${results.crisis_indicators.map(indicator => `<li>${indicator}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Recommendations
    if (results.recommendations && results.recommendations.length > 0) {
        html += `
            <div class="recommendations">
                <h4><i class="fas fa-lightbulb"></i> Personalized Recommendations:</h4>
                ${results.recommendations.map(rec => 
                    `<div class="recommendation-item">
                        <i class="fas fa-arrow-right" style="color: var(--primary); margin-right: 0.5rem;"></i>
                        ${rec}
                    </div>`
                ).join('')}
            </div>
        `;
    }
    
    // Action buttons
    html += `
        <div style="margin-top: 2rem; display: flex; gap: 1rem; flex-wrap: wrap;">
            <button class="btn btn-primary" onclick="showSection('chat')">
                <i class="fas fa-comments"></i> Continue with Chat Support
            </button>
            <button class="btn btn-success" onclick="showSection('resources')">
                <i class="fas fa-heart"></i> View Resources
            </button>
            <button class="btn btn-outline" onclick="showSection('assessment')">
                <i class="fas fa-redo"></i> Take Another Assessment
            </button>
        </div>
    `;
    
    container.innerHTML = html;
}

// =============================================================================
// CHAT FUNCTIONS
// =============================================================================

// Send chat message
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message || isProcessing) return;
    
    // Add user message to chat
    addMessageToChat(message, 'user');
    input.value = '';
    isProcessing = true;
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await apiRequest('/api/chat/message', {
            method: 'POST',
            body: JSON.stringify({
                content: message,
                message_type: 'user'
            }),
        });
        
        hideTypingIndicator();
        
        if (response.success && response.data) {
            // Add AI response to chat
            addMessageToChat(response.data.ai_response, 'ai');
            
            // Show suggested resources if available
            if (response.data.suggested_resources && response.data.suggested_resources.length > 0) {
                addResourceSuggestions(response.data.suggested_resources);
            }
            
            // Handle crisis situations
            if (response.data.requires_human_intervention) {
                showCrisisAlert();
            }
        } else {
            throw new Error(response.message || 'Failed to get response');
        }
    } catch (error) {
        hideTypingIndicator();
        console.error('Chat error:', error);
        addMessageToChat('I apologize, but I\'m having trouble responding right now. Please try again, or contact 988 if you need immediate support.', 'ai');
    } finally {
        isProcessing = false;
    }
}

// Add message to chat UI
function addMessageToChat(content, sender) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatar = sender === 'user' ? 'YOU' : 'AI';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <p>${content}</p>
            <small style="opacity: 0.7; font-size: 0.8rem;">${time}</small>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    
    // Store message for context
    chatMessages.push({ content, sender, timestamp: new Date() });
    
    scrollChatToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message ai typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <p><i class="fas fa-ellipsis-h fa-pulse"></i> Thinking...</p>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    scrollChatToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Add resource suggestions to chat
function addResourceSuggestions(resources) {
    const messagesContainer = document.getElementById('chatMessages');
    const resourceDiv = document.createElement('div');
    resourceDiv.className = 'message ai';
    
    const resourcesHtml = resources.map(resource => 
        `<li style="margin-bottom: 0.5rem;">🔹 ${resource}</li>`
    ).join('');
    
    resourceDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <p><strong>Here are some helpful resources:</strong></p>
            <ul style="margin-top: 0.5rem; padding-left: 1rem;">
                ${resourcesHtml}
            </ul>
            <small style="opacity: 0.7;">These suggestions are personalized based on our conversation.</small>
        </div>
    `;
    
    messagesContainer.appendChild(resourceDiv);
    scrollChatToBottom();
}

// Show crisis alert
function showCrisisAlert() {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'emergency-banner';
    alertDiv.style.cssText = 'position: fixed; top: 70px; left: 50%; transform: translateX(-50%); z-index: 1000; max-width: 600px;';
    
    alertDiv.innerHTML = `
        <strong>🚨 CRISIS SUPPORT NEEDED</strong><br>
        <strong>988</strong> - Crisis Lifeline | Text <strong>HOME</strong> to <strong>741741</strong>
        <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; color: white; font-size: 1.2rem;">×</button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (alertDiv.parentElement) {
            alertDiv.remove();
        }
    }, 10000);
}

// Clear chat
function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        document.getElementById('chatMessages').innerHTML = `
            <div class="message ai">
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <p>Hello! I'm your AI mental health support assistant. How can I help you today?</p>
                </div>
            </div>
        `;
        chatMessages = [];
        showNotification('Chat cleared', 'success');
    }
}

// Scroll chat to bottom
function scrollChatToBottom() {
    const messagesContainer = document.getElementById('chatMessages');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

// Assessment form handler
document.getElementById('assessmentForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const textInput = document.getElementById('assessmentText').value.trim();
    const assessmentType = document.getElementById('assessmentType').value;
    
    if (!textInput) {
        showNotification('Please share your thoughts before submitting.', 'warning');
        return;
    }
    
    if (textInput.length < 10) {
        showNotification('Please provide more detail (at least 10 characters) for a better assessment.', 'warning');
        return;
    }
    
    await processAssessment(textInput, assessmentType);
});

// Login form handler
document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value;
    
    if (!email || !password) {
        showNotification('Please fill in all fields.', 'warning');
        return;
    }
    
    await login(email, password);
});

// Register form handler
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        email: document.getElementById('registerEmail').value.trim(),
        username: document.getElementById('registerUsername').value.trim(),
        password: document.getElementById('registerPassword').value,
        first_name: document.getElementById('registerFirstName').value.trim(),
        data_sharing_consent: document.getElementById('dataConsent').checked
    };
    
    // Validation
    if (!formData.email || !formData.username || !formData.password) {
        showNotification('Please fill in all required fields.', 'warning');
        return;
    }
    
    if (formData.password.length < 8) {
        showNotification('Password must be at least 8 characters long.', 'warning');
        return;
    }
    
    if (!formData.data_sharing_consent) {
        showNotification('Please consent to data processing to create an account.', 'warning');
        return;
    }
    
    await register(formData);
});

// Chat input handler
document.getElementById('chatInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// =============================================================================
// ADDITIONAL FEATURES
// =============================================================================

// Show mental health apps
function showApps() {
    const apps = [
        { name: 'Headspace', description: 'Meditation and mindfulness', url: 'https://headspace.com' },
        { name: 'Calm', description: 'Sleep and relaxation', url: 'https://calm.com' },
        { name: 'Youper', description: 'AI emotional health assistant', url: 'https://youper.ai' },
        { name: 'Sanvello', description: 'Anxiety and depression support', url: 'https://sanvello.com' },
        { name: 'Talkspace', description: 'Online therapy platform', url: 'https://talkspace.com' },
        { name: 'BetterHelp', description: 'Professional counseling', url: 'https://betterhelp.com' }
    ];
    
    const appsHtml = apps.map(app => 
        `<div class="resource-card">
            <h4>${app.name}</h4>
            <p>${app.description}</p>
            <a href="${app.url}" target="_blank" class="btn btn-outline">Visit App</a>
        </div>`
    ).join('');
    
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
        background: rgba(0,0,0,0.8); z-index: 2000; display: flex; 
        align-items: center; justify-content: center; padding: 2rem;
    `;
    
    modal.innerHTML = `
        <div style="background: white; border-radius: var(--radius); max-width: 800px; width: 100%; max-height: 80vh; overflow-y: auto;">
            <div style="padding: 2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3>Recommended Mental Health Apps</h3>
                    <button onclick="this.closest('.modal').remove()" style="background: none; border: none; font-size: 1.5rem; cursor: pointer;">×</button>
                </div>
                <div class="resource-grid">
                    ${appsHtml}
                </div>
            </div>
        </div>
    `;
    
    modal.className = 'modal';
    document.body.appendChild(modal);
    
    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Check system health
async function checkSystemHealth() {
    try {
        const response = await apiRequest('/health');
        console.log('System Health:', response);
        return response.overall_status === 'healthy';
    } catch (error) {
        console.error('Health check failed:', error);
        showNotification('Connection to server failed. Some features may not work.', 'warning');
        return false;
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Mental Health AI Frontend Initialized');
    
    // Check system health
    await checkSystemHealth();
    
    // Initialize authentication
    initAuth();
    
    // Show home section by default
    const hash = window.location.hash.substring(1);
    if (hash && document.getElementById(hash + 'Section')) {
        showSection(hash);
    } else {
        showSection('home');
    }
    
    // Show welcome notification
    setTimeout(() => {
        showNotification('Welcome to Mental Health AI! Your privacy and wellbeing are our priority.', 'success');
    }, 1000);
});

// Handle browser back/forward
window.addEventListener('hashchange', () => {
    const hash = window.location.hash.substring(1);
    if (hash && document.getElementById(hash + 'Section')) {
        showSection(hash);
    }
});

// Handle offline/online status
window.addEventListener('online', () => {
    showNotification('Connection restored. All features are available.', 'success');
});

window.addEventListener('offline', () => {
    showNotification('You are offline. Some features may not work.', 'warning');
});

// Prevent form submission on Enter for textareas
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.tagName === 'TEXTAREA' && !e.shiftKey) {
        // Allow normal behavior for textareas
        return;
    }
});

// Auto-save assessment draft (every 5 seconds)
let assessmentDraftTimer;
const assessmentTextarea = document.getElementById('assessmentText');

assessmentTextarea?.addEventListener('input', () => {
    clearTimeout(assessmentDraftTimer);
    assessmentDraftTimer = setTimeout(() => {
        const draft = assessmentTextarea.value;
        if (draft.length > 20) {
            localStorage.setItem('assessmentDraft', draft);
        }
    }, 5000);
});

// Restore assessment draft
const savedDraft = localStorage.getItem('assessmentDraft');
if (savedDraft && assessmentTextarea) {
    assessmentTextarea.value = savedDraft;
    // Show restore notification
    setTimeout(() => {
        showNotification('Restored your previous draft.', 'success');
    }, 2000);
}

// Clear draft after successful submission
function clearAssessmentDraft() {
    localStorage.removeItem('assessmentDraft');
}

// Export functions for global access
window.showSection = showSection;
window.sendMessage = sendMessage;
window.clearChat = clearChat;
window.showApps = showApps;
window.logout = logout;