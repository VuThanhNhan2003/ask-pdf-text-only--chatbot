/**
 * Widget Loader Script
 * Paste this script on your website to embed the chatbot widget
 * 
 * Usage:
 * <script src="https://your-domain.com/widget/loader.js" 
 *         data-api-url="http://localhost:8000"
 *         data-subject="Your Subject"
 *         data-position="bottom-right"
 *         data-primary-color="#1f77b4"></script>
 */

(function() {
    'use strict';

    // Get configuration from script tag attributes
    const currentScript = document.currentScript || 
        (function() {
            const scripts = document.getElementsByTagName('script');
            return scripts[scripts.length - 1];
        })();

    const config = {
        apiUrl: currentScript.getAttribute('data-api-url') || 'http://localhost:8000',
        widgetUrl: currentScript.getAttribute('data-widget-url') || 
                   (currentScript.src.replace('loader.js', 'chatbot-widget.js')),
        subject: currentScript.getAttribute('data-subject') || null,
        modelKey: currentScript.getAttribute('data-model-key') || null,
        position: currentScript.getAttribute('data-position') || 'bottom-right',
        primaryColor: currentScript.getAttribute('data-primary-color') || '#1f77b4',
        textColor: currentScript.getAttribute('data-text-color') || '#333333',
        backgroundColor: currentScript.getAttribute('data-background-color') || '#ffffff',
        buttonText: currentScript.getAttribute('data-button-text') || '🤖',
        welcomeMessage: currentScript.getAttribute('data-welcome-message') || 'Xin chào! Tôi có thể giúp gì cho bạn?',
        placeholder: currentScript.getAttribute('data-placeholder') || 'Nhập câu hỏi của bạn...'
    };

    // Load the widget script
    function loadWidget() {
        const script = document.createElement('script');
        script.src = config.widgetUrl;
        script.async = true;
        script.onload = function() {
            // Initialize widget with configuration
            if (window.ChatbotWidget) {
                const widget = new window.ChatbotWidget({
                    apiUrl: config.apiUrl,
                    subject: config.subject,
                    modelKey: config.modelKey,
                    position: config.position,
                    primaryColor: config.primaryColor,
                    textColor: config.textColor,
                    backgroundColor: config.backgroundColor,
                    buttonText: config.buttonText,
                    welcomeMessage: config.welcomeMessage,
                    placeholder: config.placeholder
                });
                window.ragChatbotWidget = widget;
            }
        };
        script.onerror = function() {
            console.error('Failed to load chatbot widget script');
        };
        document.head.appendChild(script);
    }

    // Load widget when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadWidget);
    } else {
        loadWidget();
    }
})();
