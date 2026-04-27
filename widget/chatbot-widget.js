/**
 * RAG Chatbot Widget — Premium Edition v2.0
 * Features: Markdown, LaTeX, Syntax Highlighting, Streaming, Subject Selector,
 *           Typing Indicator, Dark/Light Mode, Copy Button, Avatars, Auto-scroll
 *
 * Dependencies loaded on demand (no bundler required):
 *   - marked.js       → Markdown
 *   - highlight.js    → Syntax highlighting
 *   - KaTeX           → LaTeX math
 *   - DOMPurify       → XSS sanitization
 */
(function () {
  'use strict';

  /* ────────────────────────────────────────────
   *  CONFIG DEFAULTS
   * ──────────────────────────────────────────── */
  const DEFAULTS = {
    apiUrl: 'https://chatbot.itp.vn:9100',
    subject: null,
    modelKey: null,
    position: 'bottom-right',        // bottom-right | bottom-left | top-right | top-left
    primaryColor: '#5b6ef5',
    welcomeMessage: 'Xin chào! 👋 Tôi là AI Assistant. Hãy chọn môn học và đặt câu hỏi nhé!',
    placeholder: 'Nhập câu hỏi của bạn…',
    buttonText: '✦',
    title: 'AI Assistant',
    subtitle: 'RAG Chatbot',
    theme: 'auto',                   // 'light' | 'dark' | 'auto'
    subjects: [],                    // Will be fetched from /subjects if empty
  };

  /* ────────────────────────────────────────────
   *  EXTERNAL LIBS (CDN) — loaded lazily
   * ──────────────────────────────────────────── */
  const LIBS = {
    marked:     'https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js',
    hljs:       'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js',
    hljsCss:    'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css',
    katex:      'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js',
    katexCss:   'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css',
    katexAuto:  'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js',
    purify:     'https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js',
  };

  const loadedLibs = new Set();

  function loadLib(url, isCSS = false) {
    if (loadedLibs.has(url)) return Promise.resolve();
    return new Promise((resolve, reject) => {
      if (isCSS) {
        const link = document.createElement('link');
        link.rel = 'stylesheet'; link.href = url;
        link.onload = resolve; link.onerror = reject;
        document.head.appendChild(link);
      } else {
        const s = document.createElement('script');
        s.src = url; s.async = true;
        s.onload = () => { loadedLibs.add(url); resolve(); };
        s.onerror = reject;
        document.head.appendChild(s);
      }
      loadedLibs.add(url);
    });
  }

  async function ensureLibs() {
    await Promise.all([
      loadLib(LIBS.hljsCss, true),
      loadLib(LIBS.katexCss, true),
    ]);
    await loadLib(LIBS.purify);
    await loadLib(LIBS.marked);
    await loadLib(LIBS.hljs);
    await loadLib(LIBS.katex);
    await loadLib(LIBS.katexAuto);
  }

  /* ────────────────────────────────────────────
   *  WIDGET CLASS
   * ──────────────────────────────────────────── */
  class ChatbotWidget {
    constructor(cfg = {}) {
      this.cfg      = { ...DEFAULTS, ...cfg };
      this.isOpen   = false;
      this.isDark   = this._detectDark();
      this.subject  = this.cfg.subject || null;   // currently selected subject
      this.subjects = this.cfg.subjects || [];     // list loaded from API
      this.libsReady = false;
      this._abortCtrl = null;

      this._injectStyles();
      this._buildDOM();
      this._bindEvents();
      this._loadSubjects();
      ensureLibs().then(() => { this.libsReady = true; });
    }

    /* ── Dark mode ── */
    _detectDark() {
      if (this.cfg.theme === 'dark') return true;
      if (this.cfg.theme === 'light') return false;
      return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
    }

    /* ──────────────────────────────────────────
     *  CSS
     * ────────────────────────────────────────── */
    _injectStyles() {
      if (document.getElementById('rcw-styles')) return;
      const p = this.cfg.primaryColor;
      const style = document.createElement('style');
      style.id = 'rcw-styles';
      style.textContent = `
        /* ── Reset ── */
        #rcw-root *, #rcw-root *::before, #rcw-root *::after {
          box-sizing: border-box; margin: 0; padding: 0;
        }

        /* ── CSS variables ── */
        #rcw-root {
          --p: ${p};
          --p-dark: color-mix(in srgb, ${p} 80%, #000);
          --p-light: color-mix(in srgb, ${p} 15%, #fff);
          --radius: 20px;
          --radius-sm: 12px;
          --shadow: 0 24px 64px rgba(0,0,0,.18), 0 4px 16px rgba(0,0,0,.1);
          --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
          /* light theme */
          --bg: #f8f9fc;
          --surface: #ffffff;
          --border: #e5e7eb;
          --text: #1f2937;
          --text-muted: #6b7280;
          --msg-user-bg: var(--p);
          --msg-user-text: #fff;
          --msg-bot-bg: #ffffff;
          --msg-bot-text: #1f2937;
          --input-bg: #ffffff;
          --header-bg: var(--p);
          --header-text: #ffffff;
          --code-bg: #1e1e2e;
          --typing-dot: #9ca3af;
        }
        #rcw-root.dark {
          --bg: #0f1117;
          --surface: #1a1d27;
          --border: #2d3148;
          --text: #e8eaf6;
          --text-muted: #8b92b8;
          --msg-user-bg: var(--p);
          --msg-user-text: #fff;
          --msg-bot-bg: #1e2235;
          --msg-bot-text: #e8eaf6;
          --input-bg: #1e2235;
          --header-bg: #1a1d27;
          --header-text: #e8eaf6;
          --code-bg: #13141f;
          --typing-dot: #4b5563;
        }

        /* ── Positioning ── */
        #rcw-root {
          position: fixed;
          z-index: 2147483647;
          font-family: var(--font);
          ${this._positionCSS()}
        }

        /* ── Launcher button ── */
        #rcw-launcher {
          width: 58px; height: 58px;
          border-radius: 50%;
          background: var(--p);
          color: #fff;
          border: none;
          cursor: pointer;
          display: flex; align-items: center; justify-content: center;
          font-size: 22px;
          box-shadow: 0 6px 24px color-mix(in srgb, ${p} 50%, transparent);
          transition: transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s;
          position: relative;
          outline: none;
        }
        #rcw-launcher:hover {
          transform: scale(1.1);
          box-shadow: 0 10px 30px color-mix(in srgb, ${p} 55%, transparent);
        }
        #rcw-launcher.hidden { display: none; }

        /* ── Unread badge ── */
        #rcw-badge {
          position: absolute; top: -4px; right: -4px;
          width: 18px; height: 18px; border-radius: 50%;
          background: #ef4444; color: #fff;
          font-size: 10px; font-weight: 700;
          display: flex; align-items: center; justify-content: center;
          display: none;
        }

        /* ── Chat window ── */
        #rcw-window {
          position: absolute;
          ${this._windowPositionCSS()}
          width: 400px;
          height: 600px;
          max-height: min(80vh, 680px);
          background: var(--bg);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          display: none;
          flex-direction: column;
          overflow: hidden;
          border: 1px solid var(--border);
          transform-origin: ${this._transformOrigin()};
        }
        #rcw-window.open {
          display: flex;
          animation: rcwPop .28s cubic-bezier(.34,1.56,.64,1);
        }
        @keyframes rcwPop {
          from { opacity: 0; transform: scale(.88); }
          to   { opacity: 1; transform: scale(1); }
        }

        /* ── Header ── */
        #rcw-header {
          background: var(--header-bg);
          color: var(--header-text);
          padding: 14px 18px;
          display: flex;
          align-items: center;
          gap: 12px;
          flex-shrink: 0;
          border-bottom: 1px solid color-mix(in srgb, var(--p) 30%, transparent);
        }
        .rcw-avatar-bot {
          width: 38px; height: 38px; border-radius: 50%;
          background: color-mix(in srgb, #fff 20%, transparent);
          display: flex; align-items: center; justify-content: center;
          font-size: 20px; flex-shrink: 0;
        }
        .rcw-header-info { flex: 1; min-width: 0; }
        .rcw-header-title { font-size: 15px; font-weight: 700; letter-spacing: -.01em; }
        .rcw-header-sub   { font-size: 11px; opacity: .75; display: flex; align-items: center; gap: 5px; }
        .rcw-status-dot   { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; display: inline-block; }
        .rcw-header-actions { display: flex; gap: 6px; }
        .rcw-hbtn {
          width: 30px; height: 30px; border-radius: 8px;
          background: color-mix(in srgb, #fff 15%, transparent);
          border: none; color: var(--header-text);
          cursor: pointer; display: flex; align-items: center; justify-content: center;
          font-size: 14px; transition: background .15s;
        }
        .rcw-hbtn:hover { background: color-mix(in srgb, #fff 25%, transparent); }

        /* ── Subject bar ── */
        #rcw-subject-bar {
          background: var(--surface);
          border-bottom: 1px solid var(--border);
          padding: 8px 14px;
          display: flex;
          align-items: center;
          gap: 8px;
          flex-shrink: 0;
          position: relative;
        }
        .rcw-subject-label { font-size: 11px; color: var(--text-muted); white-space: nowrap; }
        .rcw-subject-chip {
          display: inline-flex; align-items: center; gap: 5px;
          background: var(--p-light);
          color: var(--p);
          border: 1px solid color-mix(in srgb, var(--p) 30%, transparent);
          border-radius: 999px;
          padding: 3px 10px 3px 9px;
          font-size: 12px; font-weight: 600;
          cursor: pointer;
          transition: background .15s, border-color .15s;
          max-width: 180px;
          overflow: hidden;
        }
        #rcw-root.dark .rcw-subject-chip {
          background: color-mix(in srgb, var(--p) 20%, transparent);
        }
        .rcw-subject-chip:hover { background: color-mix(in srgb, var(--p) 20%, transparent); }
        .rcw-subject-chip.neutral {
          background: color-mix(in srgb, var(--text-muted) 12%, transparent);
          color: var(--text-muted);
          border-color: var(--border);
        }
        .rcw-chip-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .rcw-chip-x {
          font-size: 14px; line-height: 1; cursor: pointer; opacity: .6;
          transition: opacity .15s; flex-shrink: 0;
        }
        .rcw-chip-x:hover { opacity: 1; }

        /* Subject popover */
        #rcw-subject-popover {
          position: absolute;
          top: calc(100% + 6px);
          left: 14px;
          width: 320px;
          max-height: 340px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          box-shadow: 0 12px 40px rgba(0,0,0,.16);
          z-index: 10;
          display: none;
          flex-direction: column;
          overflow: hidden;
        }
        #rcw-subject-popover.open { display: flex; animation: rcwFadeDown .18s ease-out; }
        @keyframes rcwFadeDown {
          from { opacity:0; transform: translateY(-6px); }
          to   { opacity:1; transform: translateY(0); }
        }
        .rcw-subject-search {
          padding: 10px 12px;
          border-bottom: 1px solid var(--border);
          position: relative;
        }
        .rcw-subject-search input {
          width: 100%; padding: 7px 12px 7px 32px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: var(--input-bg);
          color: var(--text);
          font-size: 13px;
          outline: none;
          font-family: var(--font);
          transition: border-color .15s;
        }
        .rcw-subject-search input:focus { border-color: var(--p); }
        .rcw-subject-search::before {
          content: '🔍'; position: absolute;
          left: 22px; top: 50%; transform: translateY(-50%);
          font-size: 13px;
        }
        .rcw-subject-list {
          overflow-y: auto; max-height: 260px;
          padding: 6px;
        }
        .rcw-subject-item {
          padding: 9px 12px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 13px;
          color: var(--text);
          transition: background .12s;
          display: flex; align-items: center; gap: 8px;
        }
        .rcw-subject-item:hover { background: var(--p-light); color: var(--p); }
        #rcw-root.dark .rcw-subject-item:hover {
          background: color-mix(in srgb, var(--p) 18%, transparent);
        }
        .rcw-subject-item.active { background: var(--p-light); color: var(--p); font-weight: 600; }
        #rcw-root.dark .rcw-subject-item.active {
          background: color-mix(in srgb, var(--p) 22%, transparent);
        }
        .rcw-subject-item .rcw-si-icon { font-size: 16px; flex-shrink: 0; }
        .rcw-subject-empty { padding: 20px; text-align: center; color: var(--text-muted); font-size: 13px; }

        /* ── Messages ── */
        #rcw-messages {
          flex: 1; overflow-y: auto;
          padding: 16px;
          display: flex; flex-direction: column; gap: 14px;
          scroll-behavior: smooth;
        }
        #rcw-messages::-webkit-scrollbar { width: 4px; }
        #rcw-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

        /* Date divider */
        .rcw-date-divider {
          text-align: center; font-size: 11px; color: var(--text-muted);
          position: relative;
          display: flex; align-items: center; gap: 10px;
        }
        .rcw-date-divider::before, .rcw-date-divider::after {
          content: ''; flex: 1; height: 1px; background: var(--border);
        }

        /* Message row */
        .rcw-row {
          display: flex; align-items: flex-end; gap: 8px;
          animation: rcwMsgIn .2s ease-out;
        }
        @keyframes rcwMsgIn {
          from { opacity:0; transform: translateY(8px); }
          to   { opacity:1; transform: translateY(0); }
        }
        .rcw-row.user { flex-direction: row-reverse; }

        /* Avatar */
        .rcw-msg-avatar {
          width: 28px; height: 28px; border-radius: 50%;
          flex-shrink: 0;
          display: flex; align-items: center; justify-content: center;
          font-size: 14px;
        }
        .rcw-msg-avatar.bot { background: var(--p-light); }
        #rcw-root.dark .rcw-msg-avatar.bot { background: color-mix(in srgb, var(--p) 20%, transparent); }
        .rcw-msg-avatar.user { background: color-mix(in srgb, var(--p) 25%, transparent); }

        /* Bubble */
        .rcw-bubble {
          max-width: 78%;
          padding: 11px 14px;
          border-radius: 18px;
          font-size: 14px;
          line-height: 1.6;
          word-break: break-word;
          position: relative;
        }
        .rcw-row.user  .rcw-bubble { background: var(--msg-user-bg); color: var(--msg-user-text); border-bottom-right-radius: 4px; }
        .rcw-row.bot   .rcw-bubble { background: var(--msg-bot-bg);  color: var(--msg-bot-text);  border-bottom-left-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.07); border: 1px solid var(--border); }

        /* Markdown inside bot bubble */
        .rcw-bubble h1,.rcw-bubble h2,.rcw-bubble h3 { font-weight: 700; margin: 12px 0 6px; }
        .rcw-bubble h1 { font-size: 17px; }
        .rcw-bubble h2 { font-size: 15px; }
        .rcw-bubble h3 { font-size: 14px; }
        .rcw-bubble p  { margin: 6px 0; }
        .rcw-bubble p:first-child { margin-top: 0; }
        .rcw-bubble p:last-child  { margin-bottom: 0; }
        .rcw-bubble ul,.rcw-bubble ol { padding-left: 18px; margin: 6px 0; }
        .rcw-bubble li { margin: 3px 0; }
        .rcw-bubble a  { color: var(--p); text-decoration: underline; }
        .rcw-bubble strong { font-weight: 700; }
        .rcw-bubble em     { font-style: italic; }
        .rcw-bubble table  { border-collapse: collapse; font-size: 13px; margin: 8px 0; width: 100%; }
        .rcw-bubble th,.rcw-bubble td { border: 1px solid var(--border); padding: 5px 9px; }
        .rcw-bubble th { background: color-mix(in srgb, var(--p) 10%, transparent); font-weight: 600; }
        .rcw-bubble blockquote { border-left: 3px solid var(--p); padding-left: 10px; color: var(--text-muted); margin: 8px 0; }
        .rcw-bubble hr { border: none; border-top: 1px solid var(--border); margin: 10px 0; }

        /* Inline code */
        .rcw-bubble code:not(pre code) {
          background: var(--code-bg); color: #e879f9;
          padding: 2px 6px; border-radius: 5px; font-size: 12.5px;
          font-family: 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
        }

        /* Code block */
        .rcw-code-block {
          position: relative; margin: 10px 0;
          border-radius: 10px; overflow: hidden;
          border: 1px solid color-mix(in srgb, #fff 10%, transparent);
        }
        .rcw-code-header {
          display: flex; align-items: center; justify-content: space-between;
          background: #11121c;
          padding: 6px 12px;
          font-size: 11px; color: #6b7280;
        }
        .rcw-code-lang { text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
        .rcw-copy-btn {
          background: none; border: none; color: #6b7280;
          cursor: pointer; font-size: 11px; display: flex; align-items: center; gap: 4px;
          padding: 2px 6px; border-radius: 5px; transition: color .15s, background .15s;
        }
        .rcw-copy-btn:hover { background: rgba(255,255,255,.08); color: #e8eaf6; }
        .rcw-copy-btn.copied { color: #4ade80; }
        .rcw-code-block pre {
          margin: 0 !important; padding: 12px 14px !important;
          background: var(--code-bg) !important;
          font-size: 12.5px !important;
          overflow-x: auto;
          font-family: 'Cascadia Code', 'Fira Code', 'Courier New', monospace !important;
        }
        .rcw-code-block pre code { background: none !important; color: inherit !important; padding: 0 !important; }

        /* Sources block */
        .rcw-sources {
          margin-top: 10px; padding-top: 10px;
          border-top: 1px solid var(--border);
          font-size: 12px; color: var(--text-muted);
        }
        .rcw-sources-title { font-weight: 600; margin-bottom: 5px; }
        .rcw-source-item { padding: 2px 0; display: flex; align-items: flex-start; gap: 5px; }

        /* Timestamp */
        .rcw-time {
          font-size: 10px; color: var(--text-muted);
          margin-top: 4px;
          text-align: right;
        }
        .rcw-row.bot .rcw-time { text-align: left; }

        /* Typing indicator */
        #rcw-typing {
          display: none;
          align-items: flex-end; gap: 8px;
        }
        #rcw-typing.visible { display: flex; animation: rcwMsgIn .2s ease-out; }
        .rcw-typing-bubble {
          background: var(--msg-bot-bg); color: var(--msg-bot-text);
          border: 1px solid var(--border);
          padding: 12px 16px; border-radius: 18px; border-bottom-left-radius: 4px;
          display: flex; align-items: center; gap: 5px;
          box-shadow: 0 1px 4px rgba(0,0,0,.07);
        }
        .rcw-typing-dot {
          width: 7px; height: 7px; border-radius: 50%;
          background: var(--p);
          animation: rcwBounce 1.2s ease-in-out infinite;
          opacity: .5;
        }
        .rcw-typing-dot:nth-child(2) { animation-delay: .2s; }
        .rcw-typing-dot:nth-child(3) { animation-delay: .4s; }
        @keyframes rcwBounce {
          0%,80%,100% { transform: translateY(0); opacity: .5; }
          40%          { transform: translateY(-6px); opacity: 1; }
        }

        /* ── Input area ── */
        #rcw-input-area {
          padding: 12px 14px;
          border-top: 1px solid var(--border);
          background: var(--surface);
          display: flex; align-items: flex-end; gap: 10px;
          flex-shrink: 0;
        }
        #rcw-input {
          flex: 1;
          background: var(--input-bg);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 10px 14px;
          font-size: 14px;
          color: var(--text);
          font-family: var(--font);
          resize: none;
          outline: none;
          min-height: 42px; max-height: 120px;
          line-height: 1.5;
          transition: border-color .2s;
          overflow-y: hidden;
        }
        #rcw-input:focus { border-color: var(--p); }
        #rcw-input::placeholder { color: var(--text-muted); }
        #rcw-input:disabled { opacity: .5; }
        #rcw-send {
          width: 42px; height: 42px; border-radius: 50%;
          background: var(--p);
          border: none; color: #fff;
          cursor: pointer; flex-shrink: 0;
          display: flex; align-items: center; justify-content: center;
          font-size: 16px;
          transition: transform .2s, background .2s, opacity .2s;
          align-self: flex-end;
        }
        #rcw-send:hover:not(:disabled) { transform: scale(1.08); background: var(--p-dark); }
        #rcw-send:disabled { opacity: .45; cursor: not-allowed; transform: none; }

        /* ── Mobile ── */
        @media (max-width: 480px) {
          #rcw-window { width: 100vw; height: 100dvh; max-height: 100dvh; border-radius: 0; ${this._mobileWindowCSS()} }
        }

        /* ── KaTeX overrides ── */
        .katex-display { margin: 10px 0; overflow-x: auto; }
        .katex { font-size: 1em; }
      `;
      document.head.appendChild(style);
    }

    _positionCSS() {
      const p = this.cfg.position;
      if (p === 'bottom-left')  return 'bottom:20px;left:20px;';
      if (p === 'top-right')    return 'top:20px;right:20px;';
      if (p === 'top-left')     return 'top:20px;left:20px;';
      return 'bottom:20px;right:20px;';
    }
    _windowPositionCSS() {
      const p = this.cfg.position;
      if (p === 'bottom-left')  return 'bottom:0;left:0;';
      if (p === 'top-right')    return 'top:0;right:0;';
      if (p === 'top-left')     return 'top:0;left:0;';
      return 'bottom:0;right:0;';
    }
    _mobileWindowCSS() {
      const p = this.cfg.position;
      if (p === 'top-right' || p === 'top-left') return 'top:0!important;right:0!important;left:0!important;';
      return 'bottom:0!important;right:0!important;left:0!important;';
    }
    _transformOrigin() {
      const p = this.cfg.position;
      if (p === 'bottom-left')  return 'bottom left';
      if (p === 'top-right')    return 'top right';
      if (p === 'top-left')     return 'top left';
      return 'bottom right';
    }

    /* ──────────────────────────────────────────
     *  DOM
     * ────────────────────────────────────────── */
    _buildDOM() {
      this._root = document.createElement('div');
      this._root.id = 'rcw-root';
      if (this.isDark) this._root.classList.add('dark');

      this._root.innerHTML = `
        <!-- Launcher -->
        <button id="rcw-launcher" title="Chat với AI">
          ${this.cfg.buttonText}
          <span id="rcw-badge"></span>
        </button>

        <!-- Chat window -->
        <div id="rcw-window">

          <!-- Header -->
          <div id="rcw-header">
            <div class="rcw-avatar-bot">🤖</div>
            <div class="rcw-header-info">
              <div class="rcw-header-title">${this.cfg.title}</div>
              <div class="rcw-header-sub">
                <span class="rcw-status-dot"></span>
                <span id="rcw-status-text">Sẵn sàng</span>
              </div>
            </div>
            <div class="rcw-header-actions">
              <button class="rcw-hbtn" id="rcw-theme-btn" title="Đổi theme">🌙</button>
              <button class="rcw-hbtn" id="rcw-clear-btn" title="Xóa chat">🗑</button>
              <button class="rcw-hbtn" id="rcw-close-btn" title="Đóng">✕</button>
            </div>
          </div>

          <!-- Subject bar -->
          <div id="rcw-subject-bar">
            <span class="rcw-subject-label">Môn học:</span>
            <div id="rcw-subject-chip" class="rcw-subject-chip neutral">
              <span class="rcw-chip-name">Tất cả môn học</span>
              <span class="rcw-chip-x" id="rcw-chip-clear" style="display:none">×</span>
            </div>

            <!-- Subject popover -->
            <div id="rcw-subject-popover">
              <div class="rcw-subject-search">
                <input id="rcw-subject-search" type="text" placeholder="Tìm môn học…" autocomplete="off" />
              </div>
              <div class="rcw-subject-list" id="rcw-subject-list">
                <div class="rcw-subject-empty">Đang tải…</div>
              </div>
            </div>
          </div>

          <!-- Messages -->
          <div id="rcw-messages">
            <!-- Welcome -->
            <div class="rcw-date-divider">Hôm nay</div>
          </div>

          <!-- Typing indicator -->
          <div id="rcw-typing" style="padding:0 16px 4px;">
            <div class="rcw-msg-avatar bot">🤖</div>
            <div class="rcw-typing-bubble">
              <div class="rcw-typing-dot"></div>
              <div class="rcw-typing-dot"></div>
              <div class="rcw-typing-dot"></div>
            </div>
          </div>

          <!-- Input -->
          <div id="rcw-input-area">
            <textarea id="rcw-input" rows="1" placeholder="${this.cfg.placeholder}"></textarea>
            <button id="rcw-send">➤</button>
          </div>
        </div>
      `;

      document.body.appendChild(this._root);

      // Cache elements
      this.$launcher   = this._root.querySelector('#rcw-launcher');
      this.$window     = this._root.querySelector('#rcw-window');
      this.$messages   = this._root.querySelector('#rcw-messages');
      this.$typing     = this._root.querySelector('#rcw-typing');
      this.$input      = this._root.querySelector('#rcw-input');
      this.$send       = this._root.querySelector('#rcw-send');
      this.$chip       = this._root.querySelector('#rcw-subject-chip');
      this.$chipName   = this._root.querySelector('.rcw-chip-name');
      this.$chipClear  = this._root.querySelector('#rcw-chip-clear');
      this.$popover    = this._root.querySelector('#rcw-subject-popover');
      this.$subjList   = this._root.querySelector('#rcw-subject-list');
      this.$subjSearch = this._root.querySelector('#rcw-subject-search');
      this.$statusTxt  = this._root.querySelector('#rcw-status-text');

      // Show welcome
      this._appendBotBubble(this.cfg.welcomeMessage, false);
    }

    /* ──────────────────────────────────────────
     *  EVENTS
     * ────────────────────────────────────────── */
    _bindEvents() {
      // Launcher
      this.$launcher.addEventListener('click', () => this._toggleWindow());
      this._root.querySelector('#rcw-close-btn').addEventListener('click', () => this._closeWindow());
      this._root.querySelector('#rcw-theme-btn').addEventListener('click', () => this._toggleDark());
      this._root.querySelector('#rcw-clear-btn').addEventListener('click', () => this._clearChat());

      // Send
      this.$send.addEventListener('click', () => this._send());
      this.$input.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this._send(); }
      });
      this.$input.addEventListener('input', () => this._autoResizeInput());

      // Subject chip
      this.$chip.addEventListener('click', e => {
        if (e.target === this.$chipClear || e.target.id === 'rcw-chip-clear') return;
        this._togglePopover();
      });
      this.$chipClear.addEventListener('click', e => { e.stopPropagation(); this._selectSubject(null); });
      this.$subjSearch.addEventListener('input', () => this._filterSubjects(this.$subjSearch.value));

      // Close popover on outside click
      document.addEventListener('click', e => {
        if (!this._root.querySelector('#rcw-subject-bar').contains(e.target)) {
          this._closePopover();
        }
      });

      // Auto dark-mode follow
      if (this.cfg.theme === 'auto') {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
          this.isDark = e.matches;
          this._root.classList.toggle('dark', this.isDark);
        });
      }
    }

    /* ──────────────────────────────────────────
     *  WINDOW TOGGLE
     * ────────────────────────────────────────── */
    _toggleWindow() {
      this.isOpen ? this._closeWindow() : this._openWindow();
    }
    _openWindow() {
      this.isOpen = true;
      this.$launcher.classList.add('hidden');
      this.$window.classList.add('open');
      this._root.querySelector('#rcw-badge').style.display = 'none';
      setTimeout(() => this.$input.focus(), 120);
      this._scrollBottom();
    }
    _closeWindow() {
      this.isOpen = false;
      this.$launcher.classList.remove('hidden');
      this.$window.classList.remove('open');
      this._closePopover();
    }
    _toggleDark() {
      this.isDark = !this.isDark;
      this._root.classList.toggle('dark', this.isDark);
      this._root.querySelector('#rcw-theme-btn').textContent = this.isDark ? '☀️' : '🌙';
    }
    _clearChat() {
      this.$messages.innerHTML = '<div class="rcw-date-divider">Hôm nay</div>';
      this._appendBotBubble(this.cfg.welcomeMessage, false);
    }

    /* ──────────────────────────────────────────
     *  SUBJECT SELECTOR
     * ────────────────────────────────────────── */
    async _loadSubjects() {
      try {
        const res = await fetch(`${this.cfg.apiUrl}/subjects`);
        const data = await res.json();
        this.subjects = data.subjects || [];
        this._renderSubjectList(this.subjects);
      } catch {
        this.subjects = this.cfg.subjects || [];
        this._renderSubjectList(this.subjects);
      }
    }

    _renderSubjectList(list) {
      this.$subjList.innerHTML = '';

      // "All subjects" item
      const allItem = document.createElement('div');
      allItem.className = 'rcw-subject-item' + (this.subject === null ? ' active' : '');
      allItem.innerHTML = `<span class="rcw-si-icon">📚</span> Tất cả môn học`;
      allItem.addEventListener('click', () => this._selectSubject(null));
      this.$subjList.appendChild(allItem);

      if (!list.length) {
        const emp = document.createElement('div');
        emp.className = 'rcw-subject-empty';
        emp.textContent = 'Không tìm thấy môn học';
        this.$subjList.appendChild(emp);
        return;
      }

      // Subject emoji mapping
      const emoji = s => {
        s = (s||'').toLowerCase();
        if (s.includes('toán') || s.includes('math'))    return '📐';
        if (s.includes('vật lý') || s.includes('physics')) return '⚡';
        if (s.includes('hóa') || s.includes('chem'))     return '🧪';
        if (s.includes('sinh') || s.includes('bio'))      return '🧬';
        if (s.includes('code') || s.includes('lập trình') || s.includes('programming')) return '💻';
        if (s.includes('mạng') || s.includes('network'))  return '🌐';
        if (s.includes('cơ sở dữ liệu') || s.includes('database')) return '🗄️';
        if (s.includes('trí tuệ') || s.includes('ai'))    return '🤖';
        if (s.includes('kinh tế') || s.includes('econ'))  return '📈';
        if (s.includes('văn') || s.includes('lit'))       return '📖';
        return '📘';
      };

      list.forEach(subj => {
        const item = document.createElement('div');
        item.className = 'rcw-subject-item' + (this.subject === subj ? ' active' : '');
        item.dataset.subj = subj;
        item.innerHTML = `<span class="rcw-si-icon">${emoji(subj)}</span> ${subj}`;
        item.addEventListener('click', () => this._selectSubject(subj));
        this.$subjList.appendChild(item);
      });
    }

    _filterSubjects(q) {
      const items = this.$subjList.querySelectorAll('.rcw-subject-item');
      q = q.toLowerCase();
      items.forEach(item => {
        const name = (item.dataset.subj || '').toLowerCase();
        item.style.display = (!q || name.includes(q) || item.dataset.subj === undefined) ? '' : 'none';
      });
    }

    _selectSubject(subj) {
      this.subject = subj;
      this.$chipName.textContent = subj || 'Tất cả môn học';
      this.$chipClear.style.display = subj ? '' : 'none';
      this.$chip.classList.toggle('neutral', !subj);
      // Update active items
      this.$subjList.querySelectorAll('.rcw-subject-item').forEach(item => {
        const isAll = item.dataset.subj === undefined;
        item.classList.toggle('active', subj === null ? isAll : item.dataset.subj === subj);
      });
      this._closePopover();
    }

    _togglePopover() {
      this.$popover.classList.toggle('open');
      if (this.$popover.classList.contains('open')) {
        this.$subjSearch.value = '';
        this._filterSubjects('');
        this.$subjSearch.focus();
      }
    }
    _closePopover() { this.$popover.classList.remove('open'); }

    /* ──────────────────────────────────────────
     *  SENDING & STREAMING
     * ────────────────────────────────────────── */
    async _send() {
      const text = this.$input.value.trim();
      if (!text) return;

      this.$input.value = '';
      this.$input.style.height = 'auto';
      this._setInputEnabled(false);
      this._appendUserBubble(text);
      this._showTyping(true);
      this._setStatus('Đang trả lời…');

      // Abort any previous stream
      if (this._abortCtrl) this._abortCtrl.abort();
      this._abortCtrl = new AbortController();

      try {
        const { bubble, rawHolder } = this._appendBotBubbleStreaming();
        let raw = '';
        let done = false;

        const res = await fetch(`${this.cfg.apiUrl}/query/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: this._abortCtrl.signal,
          body: JSON.stringify({
            question: text,
            subject: this.subject,
            model_key: this.cfg.modelKey,
            use_history: true,
          }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        this._showTyping(false);

        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';

        while (true) {
          const { value, done: streamDone } = await reader.read();
          if (streamDone) break;
          buf += dec.decode(value, { stream: true });

          const parts = buf.split('\n\n');
          buf = parts.pop() ?? '';

          for (const part of parts) {
            if (!part.startsWith('data: ')) continue;
            try {
              const d = JSON.parse(part.slice(6));
              if (d.error) throw new Error(d.error);
              if (d.done)  { done = true; break; }
              if (d.chunk) {
                raw += d.chunk;
                rawHolder.textContent = raw;           // plain text while streaming
                this._scrollBottom();
              }
            } catch {}
          }
          if (done) break;
        }

        // Final render
        this._renderBotBubble(bubble, raw);

      } catch (err) {
        this._showTyping(false);
        if (err.name !== 'AbortError') {
          this._appendBotBubble(`❌ Lỗi: ${err.message}`);
        }
      }

      this._setInputEnabled(true);
      this._setStatus('Sẵn sàng');
      this.$input.focus();
    }

    /* ──────────────────────────────────────────
     *  BUBBLE HELPERS
     * ────────────────────────────────────────── */
    _appendUserBubble(text) {
      const row = document.createElement('div');
      row.className = 'rcw-row user';
      row.innerHTML = `
        <div>
          <div class="rcw-bubble">${this._esc(text)}</div>
          <div class="rcw-time">${this._timeNow()}</div>
        </div>
        <div class="rcw-msg-avatar user">👤</div>
      `;
      this.$messages.appendChild(row);
      this._scrollBottom();
    }

    /** Bot bubble: plain text while streaming */
    _appendBotBubbleStreaming() {
      const row = document.createElement('div');
      row.className = 'rcw-row bot';
      const bubble = document.createElement('div');
      bubble.className = 'rcw-bubble';
      const rawHolder = document.createElement('span');
      rawHolder.style.cssText = 'white-space:pre-wrap;';
      bubble.appendChild(rawHolder);
      const timeEl = document.createElement('div');
      timeEl.className = 'rcw-time';
      timeEl.textContent = this._timeNow();

      row.innerHTML = `<div class="rcw-msg-avatar bot">🤖</div>`;
      const wrapper = document.createElement('div');
      wrapper.appendChild(bubble);
      wrapper.appendChild(timeEl);
      row.appendChild(wrapper);

      this.$messages.appendChild(row);
      this._scrollBottom();
      return { bubble, rawHolder };
    }

    /** Replace streaming placeholder with fully-rendered markdown/math */
    _renderBotBubble(bubble, raw) {
      bubble.innerHTML = this._render(raw);
      this._highlightCode(bubble);
      this._renderMath(bubble);
      this._attachCopyBtns(bubble);
      this._formatSources(bubble);
      this._scrollBottom();
    }

    /** Standalone bot bubble (welcome, error) */
    _appendBotBubble(text, render = true) {
      const row = document.createElement('div');
      row.className = 'rcw-row bot';
      const bubble = document.createElement('div');
      bubble.className = 'rcw-bubble';
      bubble.innerHTML = render ? this._render(text) : this._esc(text);
      const timeEl = document.createElement('div');
      timeEl.className = 'rcw-time';
      timeEl.textContent = this._timeNow();
      const wrapper = document.createElement('div');
      wrapper.appendChild(bubble); wrapper.appendChild(timeEl);
      row.innerHTML = `<div class="rcw-msg-avatar bot">🤖</div>`;
      row.appendChild(wrapper);
      this.$messages.appendChild(row);
      if (render) { this._highlightCode(bubble); this._renderMath(bubble); this._attachCopyBtns(bubble); }
      this._scrollBottom();
      return bubble;
    }

    /* ──────────────────────────────────────────
     *  MARKDOWN + MATH + CODE RENDERING
     * ────────────────────────────────────────── */
    _render(raw) {
      if (!window.marked || !window.DOMPurify) {
        return `<span style="white-space:pre-wrap">${this._esc(raw)}</span>`;
      }

      // Pre-process: wrap code blocks with custom wrapper
      marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: null,
      });

      const renderer = new marked.Renderer();

      renderer.code = (code, lang) => {
        const safeLang = lang ? this._esc(lang) : 'text';
        const safeCode = this._esc(typeof code === 'object' ? code.text : code);
        return `
          <div class="rcw-code-block">
            <div class="rcw-code-header">
              <span class="rcw-code-lang">${safeLang}</span>
              <button class="rcw-copy-btn" data-code="${safeCode.replace(/"/g, '&quot;')}">
                📋 Copy
              </button>
            </div>
            <pre><code class="language-${safeLang}">${safeCode}</code></pre>
          </div>`;
      };

      renderer.codespan = (code) => {
        const text = typeof code === 'object' ? code.text : code;
        return `<code>${this._esc(text)}</code>`;
      };

      marked.use({ renderer });

      const html = marked.parse(raw);
      return DOMPurify.sanitize(html, {
        ADD_TAGS: ['div', 'span'],
        ADD_ATTR: ['class', 'data-code', 'style'],
      });
    }

    _highlightCode(container) {
      if (!window.hljs) return;
      container.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
      });
    }

    _renderMath(container) {
      if (!window.renderMathInElement) return;
      try {
        renderMathInElement(container, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
        });
      } catch {}
    }

    _attachCopyBtns(container) {
      container.querySelectorAll('.rcw-copy-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
          const code = btn.dataset.code || '';
          try {
            await navigator.clipboard.writeText(
              code.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
            );
            btn.textContent = '✅ Copied!';
            btn.classList.add('copied');
            setTimeout(() => { btn.textContent = '📋 Copy'; btn.classList.remove('copied'); }, 2000);
          } catch {}
        });
      });
    }

    /** Format "📚 **Nguồn tham khảo:**\n- …" lines into a styled block */
    _formatSources(container) {
      const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);

      for (const node of nodes) {
        if (!node.textContent.includes('📚')) continue;
        const parent = node.parentElement;
        if (!parent) continue;
        const full = parent.innerHTML;
        if (!full.includes('📚')) continue;

        const srcIdx = full.indexOf('📚');
        if (srcIdx === -1) continue;

        const before = full.slice(0, srcIdx);
        const srcRaw = full.slice(srcIdx);

        const lines = srcRaw.replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, '').split('\n').filter(Boolean);
        const title = lines[0].replace(/\*\*/g, '');
        const items = lines.slice(1).map(l => l.replace(/^[-*•]\s*/, '').trim()).filter(Boolean);

        const srcHTML = `
          <div class="rcw-sources">
            <div class="rcw-sources-title">${this._esc(title)}</div>
            ${items.map(i => `<div class="rcw-source-item">📄 <span>${this._esc(i)}</span></div>`).join('')}
          </div>`;

        parent.innerHTML = before + srcHTML;
        break;
      }
    }

    /* ──────────────────────────────────────────
     *  UI HELPERS
     * ────────────────────────────────────────── */
    _showTyping(show) {
      this.$typing.classList.toggle('visible', show);
      if (show) this._scrollBottom();
    }
    _setInputEnabled(enabled) {
      this.$input.disabled  = !enabled;
      this.$send.disabled   = !enabled;
    }
    _setStatus(txt) { this.$statusTxt.textContent = txt; }
    _scrollBottom() {
      this.$messages.scrollTop = this.$messages.scrollHeight;
    }
    _autoResizeInput() {
      this.$input.style.height = 'auto';
      this.$input.style.height = Math.min(this.$input.scrollHeight, 120) + 'px';
    }
    _timeNow() {
      return new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    }
    _esc(str) {
      return String(str ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }
  }

  /* ────────────────────────────────────────────
   *  AUTO-INIT
   * ──────────────────────────────────────────── */
  function initFromScript() {
    const s = document.querySelector('script[data-widget-init]')
           || document.querySelector('script[data-api-url]');
    if (!s) return;

    const get = k => s.getAttribute(k);
    const cfg = {
      apiUrl:         get('data-api-url')          || DEFAULTS.apiUrl,
      subject:        get('data-subject')           || null,
      modelKey:       get('data-model-key')         || null,
      position:       get('data-position')          || DEFAULTS.position,
      primaryColor:   get('data-primary-color')     || DEFAULTS.primaryColor,
      title:          get('data-title')             || DEFAULTS.title,
      subtitle:       get('data-subtitle')          || DEFAULTS.subtitle,
      buttonText:     get('data-button-text')       || DEFAULTS.buttonText,
      welcomeMessage: get('data-welcome-message')   || DEFAULTS.welcomeMessage,
      placeholder:    get('data-placeholder')       || DEFAULTS.placeholder,
      theme:          get('data-theme')             || DEFAULTS.theme,
    };
    window.ragChatbotWidget = new ChatbotWidget(cfg);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFromScript);
  } else {
    initFromScript();
  }

  // Export for manual init
  window.ChatbotWidget = ChatbotWidget;
})();