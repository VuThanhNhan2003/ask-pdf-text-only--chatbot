/**
 * RAG Chatbot Widget
 * Embeddable chat widget for AI chatbot
 * FIXED: Input Alignment & Launcher Visibility
 */

(function() {
    'use strict';

    // Widget configuration
    const defaultConfig = {
        apiUrl: 'https://chatbot.itp.vn:9100',  // Default API URL
        subject: null,  // Optional subject filter
        modelKey: null,  // Optional model key
        position: 'bottom-right',  // Widget position
        primaryColor: '#1f77b4',
        textColor: '#333333',
        backgroundColor: '#ffffff',
        buttonText: '💬 Chat',
        welcomeMessage: 'Xin chào! Tôi có thể giúp gì cho bạn?',
        placeholder: 'Nhập câu hỏi của bạn...'
    };

    class ChatbotWidget {
        constructor(config = {}) {
            this.config = { ...defaultConfig, ...config };
            this.isOpen = false;
            this.messageHistory = [];
            this.currentStream = null;
            this.init();
        }

        init() {
            this.createWidget();
            this.attachEventListeners();
        }

        createWidget() {
            // Create widget container
            this.widgetContainer = document.createElement('div');
            this.widgetContainer.id = 'rag-chatbot-widget';
            this.widgetContainer.innerHTML = `
                <div class="rag-chatbot-button" id="rag-chatbot-button">
                    <span>${this.config.buttonText}</span>
                </div>
                <div class="rag-chatbot-window" id="rag-chatbot-window">
                    <div class="rag-chatbot-header">
                        <h3>🤖 AI Chatbot</h3>
                        <button class="rag-chatbot-close" id="rag-chatbot-close">×</button>
                    </div>
                    <div class="rag-chatbot-messages" id="rag-chatbot-messages">
                        <div class="rag-chatbot-message rag-chatbot-message-assistant">
                            <div class="rag-chatbot-message-content">${this.config.welcomeMessage}</div>
                        </div>
                    </div>
                    <div class="rag-chatbot-input-container">
                        <textarea 
                            id="rag-chatbot-input" 
                            class="rag-chatbot-input" 
                            placeholder="${this.config.placeholder}"
                            rows="1"
                        ></textarea>
                        <button id="rag-chatbot-send" class="rag-chatbot-send">➤</button>
                    </div>
                </div>
            `;
            document.body.appendChild(this.widgetContainer);
            this.applyStyles();
        }

        applyStyles() {
            const style = document.createElement('style');
            style.textContent = `
                /* Reset box-sizing để tránh vỡ layout */
                #rag-chatbot-widget * {
                    box-sizing: border-box;
                }

                #rag-chatbot-widget {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    position: fixed;
                    z-index: 10000;
                    ${this.getPositionStyles()}
                }

                .rag-chatbot-button {
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: ${this.config.primaryColor};
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
                    font-size: 24px;
                    /* Đảm bảo nút không bị méo */
                    flex-shrink: 0; 
                }

                .rag-chatbot-button:hover {
                    transform: scale(1.1);
                    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
                }

                /* Class ẩn nút khi mở chat */
                .rag-chatbot-button.hidden {
                    display: none !important;
                }

                .rag-chatbot-window {
                    position: absolute;
                    bottom: 0; /* Sửa lại vị trí bottom để nó thay thế chỗ nút tròn */
                    right: 0;
                    width: 380px;
                    height: 600px;
                    max-height: 80vh; /* Tránh bị quá cao trên màn hình nhỏ */
                    background: ${this.config.backgroundColor};
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                    display: none;
                    flex-direction: column;
                    overflow: hidden;
                    animation: slideUp 0.3s ease-out;
                }

                @keyframes slideUp {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                .rag-chatbot-window.open {
                    display: flex;
                }

                .rag-chatbot-header {
                    background: ${this.config.primaryColor};
                    color: white;
                    padding: 16px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .rag-chatbot-header h3 {
                    margin: 0;
                    font-size: 18px;
                    font-weight: 600;
                }

                .rag-chatbot-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 28px;
                    cursor: pointer;
                    padding: 0;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    transition: background 0.2s;
                }

                .rag-chatbot-close:hover {
                    background: rgba(255, 255, 255, 0.2);
                }

                .rag-chatbot-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }

                .rag-chatbot-message {
                    display: flex;
                    max-width: 80%;
                    animation: fadeIn 0.3s;
                }

                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                .rag-chatbot-message-user {
                    align-self: flex-end;
                }

                .rag-chatbot-message-assistant {
                    align-self: flex-start;
                }

                .rag-chatbot-message-content {
                    padding: 12px 16px;
                    border-radius: 18px;
                    word-wrap: break-word;
                    line-height: 1.5;
                }

                .rag-chatbot-message-user .rag-chatbot-message-content {
                    background: ${this.config.primaryColor};
                    color: white;
                }

                .rag-chatbot-message-assistant .rag-chatbot-message-content {
                    background: #f0f0f0;
                    color: ${this.config.textColor};
                }

                /* ----- PHẦN SỬA LỖI INPUT ----- */
                .rag-chatbot-input-container {
                    padding: 16px;
                    border-top: 1px solid #e0e0e0;
                    display: flex;
                    gap: 10px;
                    align-items: center; /* Căn giữa theo chiều dọc */
                    background: #fff;
                }

                .rag-chatbot-input {
                    flex: 1;
                    border: 1px solid #e0e0e0;
                    border-radius: 20px;
                    padding: 10px 15px;
                    font-size: 14px;
                    resize: none;
                    max-height: 100px;
                    font-family: inherit;
                    outline: none;
                    transition: border-color 0.2s;
                    margin: 0; /* Xóa margin mặc định */
                    line-height: 1.4;
                    min-height: 42px; /* Chiều cao tối thiểu khớp với nút gửi */
                    /* Ẩn thanh cuộn nếu ít text */
                    overflow-y: hidden;
                }

                .rag-chatbot-input:focus {
                    border-color: ${this.config.primaryColor};
                }

                .rag-chatbot-send {
                    width: 42px; /* Khớp với chiều cao input */
                    height: 42px;
                    border-radius: 50%;
                    background: ${this.config.primaryColor};
                    color: white;
                    border: none;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    transition: transform 0.2s, background 0.2s;
                    flex-shrink: 0; /* Không cho nút bị bóp méo */
                    padding: 0;
                    margin: 0;
                }

                .rag-chatbot-send:hover:not(:disabled) {
                    transform: scale(1.05);
                    background: ${this.darkenColor(this.config.primaryColor, 10)};
                }

                .rag-chatbot-send:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                /* Scrollbar styling */
                .rag-chatbot-messages::-webkit-scrollbar {
                    width: 6px;
                }
                .rag-chatbot-messages::-webkit-scrollbar-track {
                    background: #f1f1f1;
                }
                .rag-chatbot-messages::-webkit-scrollbar-thumb {
                    background: #ccc;
                    border-radius: 3px;
                }

                /* Mobile responsive */
                @media (max-width: 480px) {
                    .rag-chatbot-window {
                        width: 100vw;
                        height: 100vh;
                        max-height: 100vh;
                        bottom: 0 !important;
                        right: 0 !important;
                        left: 0 !important;
                        border-radius: 0;
                    }
                    .rag-chatbot-button {
                        bottom: 20px;
                        right: 20px;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        getPositionStyles() {
            const positions = {
                'bottom-right': 'bottom: 20px; right: 20px;',
                'bottom-left': 'bottom: 20px; left: 20px;',
                'top-right': 'top: 20px; right: 20px;',
                'top-left': 'top: 20px; left: 20px;'
            };
            return positions[this.config.position] || positions['bottom-right'];
        }

        darkenColor(color, percent) {
            const num = parseInt(color.replace("#", ""), 16);
            const amt = Math.round(2.55 * percent);
            const R = Math.max(0, Math.min(255, (num >> 16) + amt));
            const G = Math.max(0, Math.min(255, (num >> 8 & 0x00FF) + amt));
            const B = Math.max(0, Math.min(255, (num & 0x0000FF) + amt));
            return "#" + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1);
        }

        attachEventListeners() {
            const button = document.getElementById('rag-chatbot-button');
            const closeBtn = document.getElementById('rag-chatbot-close');
            const sendBtn = document.getElementById('rag-chatbot-send');
            const input = document.getElementById('rag-chatbot-input');

            button.addEventListener('click', () => this.toggleWidget());
            closeBtn.addEventListener('click', () => this.closeWidget());
            sendBtn.addEventListener('click', () => this.sendMessage());
            
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            // Auto-resize textarea
            input.addEventListener('input', () => {
                input.style.height = 'auto';
                input.style.height = Math.min(input.scrollHeight, 100) + 'px';
                
                // Hiện thanh cuộn nếu cao hơn 100px
                if(input.scrollHeight > 100) {
                    input.style.overflowY = 'auto';
                } else {
                    input.style.overflowY = 'hidden';
                }
            });
        }

        toggleWidget() {
            this.isOpen = !this.isOpen;
            const windowEl = document.getElementById('rag-chatbot-window');
            const buttonEl = document.getElementById('rag-chatbot-button');
            
            if (this.isOpen) {
                windowEl.classList.add('open');
                buttonEl.classList.add('hidden'); // <--- Ẩn nút tròn
                setTimeout(() => document.getElementById('rag-chatbot-input').focus(), 100);
            } else {
                windowEl.classList.remove('open');
                buttonEl.classList.remove('hidden'); // <--- Hiện nút tròn
            }
        }

        closeWidget() {
            this.isOpen = false;
            document.getElementById('rag-chatbot-window').classList.remove('open');
            document.getElementById('rag-chatbot-button').classList.remove('hidden'); // <--- Hiện lại nút tròn khi đóng
        }

        addMessage(content, role = 'assistant') {
            const messagesContainer = document.getElementById('rag-chatbot-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `rag-chatbot-message rag-chatbot-message-${role}`;
            messageDiv.innerHTML = `
                <div class="rag-chatbot-message-content">${this.escapeHtml(content)}</div>
            `;
            messagesContainer.appendChild(messageDiv);
            this.scrollToBottom();
        }

        addStreamingMessage() {
            const messagesContainer = document.getElementById('rag-chatbot-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'rag-chatbot-message rag-chatbot-message-assistant';
            messageDiv.innerHTML = `
                <div class="rag-chatbot-message-content"></div>
            `;
            messagesContainer.appendChild(messageDiv);
            this.scrollToBottom();
            return messageDiv.querySelector('.rag-chatbot-message-content');
        }

        updateStreamingMessage(element, chunk) {
            element.textContent += chunk;
            this.scrollToBottom();
        }

        async sendMessage() {
            const input = document.getElementById('rag-chatbot-input');
            const sendBtn = document.getElementById('rag-chatbot-send');
            const message = input.value.trim();

            if (!message) return;

            // Disable input
            input.disabled = true;
            sendBtn.disabled = true;

            // Add user message
            this.addMessage(message, 'user');
            this.messageHistory.push({ role: 'user', content: message });
            input.value = '';
            input.style.height = 'auto'; // Reset height sau khi gửi

            // Create streaming message element
            const streamingElement = this.addStreamingMessage();

            try {
                // Call streaming API
                const response = await fetch(`${this.config.apiUrl}/query/stream`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: message,
                        subject: this.config.subject,
                        model_key: this.config.modelKey,
                        use_history: true
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.chunk) {
                                    this.updateStreamingMessage(streamingElement, data.chunk);
                                }
                                if (data.done) {
                                    break;
                                }
                                if (data.error) {
                                    throw new Error(data.error);
                                }
                            } catch (e) {
                                console.error('Error parsing SSE data:', e);
                            }
                        }
                    }
                }

                // Process remaining buffer
                if (buffer) {
                    const lines = buffer.split('\n\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.chunk) {
                                    this.updateStreamingMessage(streamingElement, data.chunk);
                                }
                            } catch (e) {
                                console.error('Error parsing SSE data:', e);
                            }
                        }
                    }
                }

                // Save assistant message to history
                const fullResponse = streamingElement.textContent;
                this.messageHistory.push({ role: 'assistant', content: fullResponse });

            } catch (error) {
                console.error('Error sending message:', error);
                streamingElement.textContent = `❌ Lỗi: ${error.message}`;
            } finally {
                // Re-enable input
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
            }
        }

        scrollToBottom() {
            const messagesContainer = document.getElementById('rag-chatbot-messages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    }

    // Auto-initialize if script is loaded with data attributes
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initWidget);
    } else {
        initWidget();
    }

    function initWidget() {
        const script = document.querySelector('script[data-widget-init]');
        if (script) {
            const config = {
                apiUrl: script.getAttribute('data-api-url') || defaultConfig.apiUrl,
                subject: script.getAttribute('data-subject') || null,
                modelKey: script.getAttribute('data-model-key') || null,
                position: script.getAttribute('data-position') || defaultConfig.position,
                primaryColor: script.getAttribute('data-primary-color') || defaultConfig.primaryColor,
                textColor: script.getAttribute('data-text-color') || defaultConfig.textColor,
                backgroundColor: script.getAttribute('data-background-color') || defaultConfig.backgroundColor,
                buttonText: script.getAttribute('data-button-text') || defaultConfig.buttonText,
                welcomeMessage: script.getAttribute('data-welcome-message') || defaultConfig.welcomeMessage,
                placeholder: script.getAttribute('data-placeholder') || defaultConfig.placeholder
            };
            window.ragChatbotWidget = new ChatbotWidget(config);
        }
    }

    // Export for manual initialization
    window.ChatbotWidget = ChatbotWidget;
})();