/**
 * Widget Loader v2 — auto-loads chatbot-widget.js and initializes
 *
 * Usage:
 *   <script src="https://your-domain.com/widget/loader.js"
 *           data-api-url="https://api.example.com"
 *           data-primary-color="#5b6ef5"
 *           data-title="AI Assistant"
 *           data-theme="auto"></script>
 */
(function () {
  'use strict';

  const currentScript =
    document.currentScript ||
    (function () {
      const s = document.getElementsByTagName('script');
      return s[s.length - 1];
    })();

  const widgetUrl =
    currentScript.getAttribute('data-widget-url') ||
    currentScript.src.replace('loader.js', 'chatbot-widget.js');

  const cfg = {
    apiUrl:         currentScript.getAttribute('data-api-url')          || 'http://localhost:9100',
    subject:        currentScript.getAttribute('data-subject')           || null,
    modelKey:       currentScript.getAttribute('data-model-key')         || null,
    position:       currentScript.getAttribute('data-position')          || 'bottom-right',
    primaryColor:   currentScript.getAttribute('data-primary-color')     || '#5b6ef5',
    title:          currentScript.getAttribute('data-title')             || 'AI Assistant',
    subtitle:       currentScript.getAttribute('data-subtitle')          || 'RAG Chatbot',
    buttonText:     currentScript.getAttribute('data-button-text')       || '✦',
    welcomeMessage: currentScript.getAttribute('data-welcome-message')   || null,
    placeholder:    currentScript.getAttribute('data-placeholder')       || null,
    theme:          currentScript.getAttribute('data-theme')             || 'auto',
  };

  // Remove null values so widget uses its own defaults
  Object.keys(cfg).forEach(k => cfg[k] === null && delete cfg[k]);

  function load() {
    if (window.ChatbotWidget) {
      window.ragChatbotWidget = new window.ChatbotWidget(cfg);
      return;
    }
    const s = document.createElement('script');
    s.src = widgetUrl;
    s.async = true;
    s.onload = () => {
      if (window.ChatbotWidget) {
        window.ragChatbotWidget = new window.ChatbotWidget(cfg);
      }
    };
    s.onerror = () => console.error('[RCW] Failed to load widget from', widgetUrl);
    document.head.appendChild(s);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', load);
  } else {
    load();
  }
})();