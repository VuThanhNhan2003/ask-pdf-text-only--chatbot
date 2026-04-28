/**
 * RAG Chatbot Widget — v3.0 "Obsidian"
 * ──────────────────────────────────────
 * Features:
 *  • KaTeX math rendering (inline $...$ and block $$...$$)
 *  • marked.js Markdown with custom renderer
 *  • highlight.js syntax highlighting (atom-one-dark palette)
 *  • Streaming response with smooth token-by-token display
 *  • Subject selector (chip + searchable popover)
 *  • Full dark / light mode
 *  • Copy button for code blocks & math
 *  • RAG citation block styling
 *  • No per-message user avatar (bot avatar only in header)
 *  • DOMPurify XSS sanitization
 *
 * Zero build-step — drop one <script> tag and go.
 */
(function () {
  'use strict';

  /* ═══════════════════════════════════════════
   * 1. CONFIG DEFAULTS
   * ═══════════════════════════════════════════ */
  const DEFAULTS = {
    apiUrl:         'https://chatbot.itp.vn:9100',
    subject:        null,
    modelKey:       null,
    position:       'bottom-right',   // bottom-right | bottom-left | top-right | top-left
    primaryColor:   '#4f6ef7',
    accentColor:    '#818cf8',
    welcomeMessage: 'Xin chào! Tôi là Smart Assistant. Chọn môn học và đặt câu hỏi để bắt đầu.',
    placeholder:    'Nhập câu hỏi…',
    title:          'Smart Assistant',
    subtitle:       'RAG · AI',
    theme:          'auto',           // 'light' | 'dark' | 'auto'
    subjects:       [],
    botEmoji:       '✦',
    launchEmoji:    '✦',
  };

  /* ═══════════════════════════════════════════
   * 2. CDN LIBRARY URLS
   * ═══════════════════════════════════════════ */
  const LIBS = {
    markedJs:     'https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js',
    hljsJs:       'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js',
    hljsCss:      'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css',
    katexJs:      'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js',
    katexCss:     'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css',
    katexAutoJs:  'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js',
    purifyJs:     'https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js',
  };

  const _loaded = new Set();
  function _loadAsset(url, isCSS = false) {
    if (_loaded.has(url)) return Promise.resolve();
    _loaded.add(url);
    return new Promise((res, rej) => {
      if (isCSS) {
        const l = document.createElement('link');
        l.rel = 'stylesheet'; l.href = url;
        l.onload = res; l.onerror = rej;
        document.head.appendChild(l);
      } else {
        const s = document.createElement('script');
        s.src = url; s.async = true;
        s.onload = res; s.onerror = rej;
        document.head.appendChild(s);
      }
    });
  }

  async function _loadAllLibs() {
    await Promise.all([
      _loadAsset(LIBS.hljsCss, true),
      _loadAsset(LIBS.katexCss, true),
    ]);
    await _loadAsset(LIBS.purifyJs);
    await _loadAsset(LIBS.markedJs);
    await _loadAsset(LIBS.hljsJs);
    await _loadAsset(LIBS.katexJs);
    await _loadAsset(LIBS.katexAutoJs);
  }

  /* ═══════════════════════════════════════════
   * 3. UTILITY
   * ═══════════════════════════════════════════ */
  const _esc = s => String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');

  const _now = () => new Date().toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit'});

  /* ═══════════════════════════════════════════
   * 4. CSS INJECTION
   * ═══════════════════════════════════════════ */
  function _injectCSS(p, accent) {
    if (document.getElementById('rcw-v3-styles')) return;
    const el = document.createElement('style');
    el.id = 'rcw-v3-styles';
    el.textContent = `
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Root & Variables ── */
#rcw-v3-root {
  --p:          ${p};
  --a:          ${accent};
  --p10:        color-mix(in srgb,${p} 10%,transparent);
  --p20:        color-mix(in srgb,${p} 20%,transparent);
  --p30:        color-mix(in srgb,${p} 30%,transparent);
  --pdk:        color-mix(in srgb,${p} 80%,#000 20%);
  --radius:     22px;
  --rsm:        14px;
  --shadow:     0 32px 80px rgba(0,0,0,.22), 0 8px 24px rgba(0,0,0,.12);
  --font:       'IBM Plex Sans', system-ui, sans-serif;
  --mono:       'IBM Plex Mono', 'Fira Code', monospace;
  font-family:  var(--font);

  /* Light palette */
  --bg:         #f5f6fa;
  --surface:    #ffffff;
  --surface2:   #f0f1f7;
  --border:     #e4e6f0;
  --text:       #1a1d2e;
  --text2:      #5a5f7d;
  --text3:      #9ca3c0;
  --user-bg:    var(--p);
  --user-txt:   #ffffff;
  --bot-bg:     #ffffff;
  --bot-txt:    #1a1d2e;
  --input-bg:   #ffffff;
  --hdr-bg:     #ffffff;
  --hdr-txt:    #1a1d2e;
  --hdr-sub:    #7c82a8;
  --code-bg:    #282c34;
  --code-hdr:   #21252b;
}
#rcw-v3-root.dark {
  --bg:         #0e0f1a;
  --surface:    #161827;
  --surface2:   #1e2038;
  --border:     #252842;
  --text:       #dde1f5;
  --text2:      #8990b8;
  --text3:      #4b5278;
  --user-bg:    var(--p);
  --user-txt:   #ffffff;
  --bot-bg:     #161827;
  --bot-txt:    #dde1f5;
  --input-bg:   #161827;
  --hdr-bg:     #0e0f1a;
  --hdr-txt:    #dde1f5;
  --hdr-sub:    #5a6090;
  --code-bg:    #1a1d2b;
  --code-hdr:   #141622;
}

/* ── Resets ── */
#rcw-v3-root *, #rcw-v3-root *::before, #rcw-v3-root *::after {
  box-sizing: border-box; margin: 0; padding: 0;
}

/* ── Positioning ── */
#rcw-v3-root {
  position: fixed;
  z-index: 2147483640;
  font-size: 14px;
}

/* ── Launcher ── */
#rcw-launcher {
  width: 56px; height: 56px;
  border-radius: 50%;
  background: var(--p);
  color: #fff;
  border: none;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; font-weight: 700;
  box-shadow: 0 6px 28px var(--p30), 0 2px 8px rgba(0,0,0,.15);
  transition: transform .25s cubic-bezier(.34,1.56,.64,1), box-shadow .2s;
  outline: none;
  position: relative;
  font-family: var(--font);
}
#rcw-launcher:hover {
  transform: scale(1.12);
  box-shadow: 0 10px 36px var(--p30);
}
#rcw-launcher.rcw-hidden { display: none; }

#rcw-badge {
  position: absolute; top: -3px; right: -3px;
  width: 17px; height: 17px; border-radius: 50%;
  background: #ef4444; color: #fff;
  font-size: 9px; font-weight: 700;
  display: none; align-items: center; justify-content: center;
  border: 2px solid #fff;
}

/* ── Chat window ── */
#rcw-window {
  display: none;
  flex-direction: column;
  background: var(--bg);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  overflow: hidden;
  width: 460px;
  height: 620px;
  max-height: min(85vh, 680px);
  transform-origin: bottom right;
}
#rcw-window.rcw-open {
  display: flex;
  animation: rcwSlideIn .28s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes rcwSlideIn {
  from { opacity: 0; transform: scale(.88) translateY(12px); }
  to   { opacity: 1; transform: scale(1)  translateY(0); }
}

/* ── Header ── */
#rcw-header {
  background: var(--hdr-bg);
  border-bottom: 1px solid var(--border);
  padding: 14px 16px;
  display: flex; align-items: center; gap: 12px;
  flex-shrink: 0;
}
.rcw-hdr-avatar {
  width: 36px; height: 36px; border-radius: 10px;
  background: var(--p);
  color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; font-weight: 700;
  flex-shrink: 0;
  letter-spacing: -.02em;
}
.rcw-hdr-info { flex: 1; min-width: 0; }
.rcw-hdr-title {
  font-size: 14px; font-weight: 600;
  color: var(--hdr-txt);
  letter-spacing: -.01em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rcw-hdr-sub {
  display: flex; align-items: center; gap: 6px;
  margin-top: 2px;
}
.rcw-status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #22c55e;
  animation: rcwPulse 2.5s ease-in-out infinite;
  flex-shrink: 0;
}
@keyframes rcwPulse {
  0%,100% { opacity: 1; }
  50%      { opacity: .4; }
}
.rcw-hdr-sub-txt {
  font-size: 11px; color: var(--hdr-sub); font-weight: 400;
  letter-spacing: .01em;
}
.rcw-hdr-actions { display: flex; gap: 4px; flex-shrink: 0; }
.rcw-hbtn {
  width: 30px; height: 30px; border-radius: 8px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text2);
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
  transition: background .15s, color .15s, border-color .15s;
}
.rcw-hbtn:hover { background: var(--surface2); color: var(--text); border-color: var(--p30); }

/* ── Subject bar ── */
#rcw-subj-bar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 8px 14px;
  display: flex; align-items: center; gap: 8px;
  flex-shrink: 0;
  position: relative;
}
.rcw-subj-label {
  font-size: 10px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--text3);
  white-space: nowrap; flex-shrink: 0;
}
.rcw-chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px; font-weight: 500;
  cursor: pointer;
  transition: all .15s;
  max-width: 220px;
  overflow: hidden;
  border: 1.5px solid;
  white-space: nowrap;
  user-select: none;
  font-family: var(--font);
}
.rcw-chip.active {
  background: var(--p10);
  border-color: var(--p30);
  color: var(--p);
}
.rcw-chip.neutral {
  background: var(--surface2);
  border-color: var(--border);
  color: var(--text2);
}
.rcw-chip:hover { border-color: var(--p); color: var(--p); background: var(--p10); }
.rcw-chip-name { overflow: hidden; text-overflow: ellipsis; }
.rcw-chip-x {
  font-size: 14px; opacity: .5; flex-shrink: 0;
  line-height: 1; transition: opacity .15s;
  cursor: pointer;
}
.rcw-chip-x:hover { opacity: 1; }

/* Subject popover */
#rcw-subj-popover {
  position: absolute;
  top: calc(100% + 6px);
  left: 14px;
  width: 300px;
  max-height: 320px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--rsm);
  box-shadow: 0 16px 48px rgba(0,0,0,.18), 0 4px 12px rgba(0,0,0,.08);
  z-index: 10;
  display: none; flex-direction: column; overflow: hidden;
}
#rcw-subj-popover.rcw-open {
  display: flex;
  animation: rcwDropIn .18s ease-out both;
}
@keyframes rcwDropIn {
  from { opacity: 0; transform: translateY(-6px) scale(.97); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
.rcw-pop-search {
  padding: 10px;
  border-bottom: 1px solid var(--border);
}
.rcw-pop-search input {
  width: 100%; padding: 7px 11px;
  border: 1.5px solid var(--border);
  border-radius: 9px;
  background: var(--input-bg);
  color: var(--text);
  font-size: 12.5px; font-family: var(--font);
  outline: none;
  transition: border-color .15s;
}
.rcw-pop-search input:focus { border-color: var(--p); }
.rcw-pop-search input::placeholder { color: var(--text3); }
.rcw-pop-list {
  overflow-y: auto; flex: 1;
  padding: 6px;
  scrollbar-width: thin; scrollbar-color: var(--border) transparent;
}
.rcw-pop-item {
  padding: 8px 11px;
  border-radius: 9px;
  cursor: pointer;
  font-size: 13px; color: var(--text);
  transition: background .12s, color .12s;
  display: flex; align-items: center; gap: 8px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rcw-pop-item:hover  { background: var(--p10); color: var(--p); }
.rcw-pop-item.active { background: var(--p10); color: var(--p); font-weight: 600; }
.rcw-pop-empty {
  padding: 24px 16px; text-align: center;
  color: var(--text3); font-size: 12.5px;
}

/* ── Messages ── */
#rcw-msgs {
  flex: 1; overflow-y: auto;
  padding: 16px 14px;
  display: flex; flex-direction: column; gap: 4px;
  scroll-behavior: smooth;
  scrollbar-width: thin; scrollbar-color: var(--border) transparent;
}
#rcw-msgs::-webkit-scrollbar { width: 4px; }
#rcw-msgs::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

.rcw-divider {
  display: flex; align-items: center; gap: 10px;
  margin: 8px 0;
  font-size: 10px; color: var(--text3); font-weight: 500;
  letter-spacing: .06em; text-transform: uppercase;
}
.rcw-divider::before, .rcw-divider::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}

/* ── Message groups ── */
.rcw-row {
  display: flex;
  align-items: flex-end;
  gap: 0;
  animation: rcwMsgIn .2s ease-out both;
}
@keyframes rcwMsgIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.rcw-row.rcw-user { flex-direction: row-reverse; margin-left: 32px; margin-top: 2px; margin-bottom: 2px; }
.rcw-row.rcw-bot  { flex-direction: row; margin-right: 32px; margin-top: 2px; margin-bottom: 2px; }
.rcw-row.rcw-bot.rcw-group-start  { margin-top: 12px; }
.rcw-row.rcw-user.rcw-group-start { margin-top: 12px; }

/* ── Bubbles ── */
.rcw-bubble {
  padding: 11px 15px;
  border-radius: 18px;
  font-size: 13.5px;
  line-height: 1.62;
  word-break: break-word;
  max-width: 100%;
  position: relative;
}
.rcw-row.rcw-user  .rcw-bubble {
  background: var(--user-bg);
  color: var(--user-txt);
  border-radius: 18px 18px 4px 18px;
  box-shadow: 0 2px 12px var(--p20);
}
.rcw-row.rcw-bot .rcw-bubble {
  background: var(--bot-bg);
  color: var(--bot-txt);
  border-radius: 18px 18px 18px 4px;
  border: 1px solid var(--border);
  box-shadow: 0 1px 6px rgba(0,0,0,.06);
}

/* Timestamp */
.rcw-meta {
  font-size: 10px; color: var(--text3);
  padding: 2px 4px 6px;
  font-weight: 400;
}
.rcw-row.rcw-user .rcw-meta { text-align: right; }
.rcw-row.rcw-bot  .rcw-meta { text-align: left; }

/* ── Markdown inside bot bubbles ── */
.rcw-bubble h1, .rcw-bubble h2, .rcw-bubble h3, .rcw-bubble h4 {
  font-weight: 600; margin: 14px 0 6px;
  line-height: 1.3; letter-spacing: -.01em;
}
.rcw-bubble h1 { font-size: 17px; }
.rcw-bubble h2 { font-size: 15px; }
.rcw-bubble h3 { font-size: 14px; }
.rcw-bubble h4 { font-size: 13.5px; }
.rcw-bubble h1:first-child, .rcw-bubble h2:first-child,
.rcw-bubble h3:first-child, .rcw-bubble h4:first-child { margin-top: 0; }
.rcw-bubble p { margin: 6px 0; }
.rcw-bubble p:first-child { margin-top: 0; }
.rcw-bubble p:last-child  { margin-bottom: 0; }
.rcw-bubble ul, .rcw-bubble ol { padding-left: 20px; margin: 6px 0; }
.rcw-bubble li { margin: 4px 0; }
.rcw-bubble a  { color: var(--p); text-decoration: underline; text-underline-offset: 2px; }
.rcw-bubble strong { font-weight: 600; }
.rcw-bubble em { font-style: italic; }
.rcw-bubble table { border-collapse: collapse; font-size: 12.5px; margin: 10px 0; width: 100%; }
.rcw-bubble th, .rcw-bubble td { border: 1px solid var(--border); padding: 6px 10px; }
.rcw-bubble th { background: var(--surface2); font-weight: 600; font-size: 12px; letter-spacing: .03em; }
.rcw-bubble blockquote {
  border-left: 3px solid var(--p);
  padding: 6px 12px; margin: 10px 0;
  color: var(--text2); background: var(--p10);
  border-radius: 0 8px 8px 0;
}
.rcw-bubble hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

/* Inline code */
.rcw-bubble code:not(pre code) {
  font-family: var(--mono);
  font-size: 12.5px;
  background: var(--surface2);
  color: #e06c75;
  padding: 1px 6px; border-radius: 5px;
  border: 1px solid var(--border);
}
.rcw-row.rcw-user .rcw-bubble code:not(pre code) {
  background: rgba(255,255,255,.18);
  color: #fff;
  border-color: rgba(255,255,255,.2);
}

/* ── Code blocks ── */
.rcw-code-wrap {
  margin: 10px 0;
  border-radius: 11px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 4px 16px rgba(0,0,0,.24);
}
.rcw-code-wrap:first-child { margin-top: 0; }
.rcw-code-wrap:last-child  { margin-bottom: 0; }
.rcw-code-hdr {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--code-hdr);
  padding: 6px 14px;
}
.rcw-code-lang {
  font-family: var(--mono);
  font-size: 10px; font-weight: 500;
  letter-spacing: .1em; text-transform: uppercase;
  color: #6b7699;
}
.rcw-copy-btn {
  background: none; border: none;
  color: #6b7699; cursor: pointer;
  font-size: 11px; font-weight: 500;
  display: flex; align-items: center; gap: 4px;
  padding: 3px 8px; border-radius: 5px;
  transition: color .15s, background .15s;
  font-family: var(--font);
}
.rcw-copy-btn:hover { background: rgba(255,255,255,.07); color: #c0c7e8; }
.rcw-copy-btn.rcw-copied { color: #4ade80; }
.rcw-code-wrap pre {
  margin: 0 !important; padding: 14px 16px !important;
  background: var(--code-bg) !important;
  font-family: var(--mono) !important;
  font-size: 12.5px !important; line-height: 1.6 !important;
  overflow-x: auto;
  scrollbar-width: thin; scrollbar-color: #3a3f58 transparent;
}
.rcw-code-wrap pre code {
  background: none !important;
  padding: 0 !important;
  border: none !important;
  font-family: var(--mono) !important;
  font-size: inherit !important;
  color: #abb2bf !important; /* atom-one-dark base text */
}

/* ── KaTeX / Math ── */
.rcw-bubble .katex-display {
  margin: 14px 0;
  overflow-x: auto; overflow-y: hidden;
  text-align: center;
  padding: 2px 0;
}
.rcw-bubble .katex-display:first-child { margin-top: 0; }
.rcw-bubble .katex-display:last-child  { margin-bottom: 0; }
.rcw-bubble .katex { font-size: 1.08em; }
.rcw-math-copy-btn {
  display: block;
  margin: -8px auto 10px;
  background: none; border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text3); cursor: pointer;
  font-size: 10px; padding: 2px 8px;
  font-family: var(--font);
  transition: color .15s, border-color .15s;
}
.rcw-math-copy-btn:hover { color: var(--p); border-color: var(--p); }

/* ── Sources / Citation ── */
.rcw-sources {
  margin-top: 12px;
  padding: 10px 13px;
  background: var(--surface2);
  border-radius: 10px;
  border: 1px solid var(--border);
}
.rcw-sources-hdr {
  display: flex; align-items: center; gap: 6px;
  font-size: 10.5px; font-weight: 600; color: var(--text3);
  letter-spacing: .06em; text-transform: uppercase;
  margin-bottom: 7px;
}
.rcw-source-item {
  display: flex; align-items: flex-start; gap: 6px;
  font-size: 12px; color: var(--text2);
  padding: 3px 0;
  line-height: 1.45;
}
.rcw-source-dot {
  width: 4px; height: 4px; border-radius: 50%;
  background: var(--p);
  margin-top: 6px; flex-shrink: 0;
}

/* ── Typing indicator ── */
#rcw-typing {
  display: none;
  align-items: center; gap: 0;
  padding: 0 14px 6px;
  margin-right: 32px;
}
#rcw-typing.rcw-vis { display: flex; animation: rcwMsgIn .2s ease-out both; }
.rcw-typing-bubble {
  background: var(--bot-bg);
  border: 1px solid var(--border);
  padding: 10px 14px;
  border-radius: 18px 18px 18px 4px;
  display: flex; align-items: center; gap: 5px;
  box-shadow: 0 1px 6px rgba(0,0,0,.06);
}
.rcw-tdot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--p);
  animation: rcwBounce 1.3s ease-in-out infinite;
  opacity: .4;
}
.rcw-tdot:nth-child(2) { animation-delay: .18s; }
.rcw-tdot:nth-child(3) { animation-delay: .36s; }
@keyframes rcwBounce {
  0%,100% { transform: translateY(0); opacity: .4; }
  40%      { transform: translateY(-5px); opacity: 1; }
}

/* ── Input area ── */
#rcw-input-area {
  padding: 10px 14px 12px;
  border-top: 1px solid var(--border);
  background: var(--surface);
  display: flex; align-items: flex-end; gap: 9px;
  flex-shrink: 0;
}
#rcw-input {
  flex: 1;
  background: var(--input-bg);
  border: 1.5px solid var(--border);
  border-radius: 14px;
  padding: 9px 14px;
  font-size: 13.5px; font-family: var(--font);
  color: var(--text);
  resize: none; outline: none;
  min-height: 40px; max-height: 130px;
  line-height: 1.5;
  transition: border-color .2s, box-shadow .2s;
  overflow-y: hidden;
}
#rcw-input:focus {
  border-color: var(--p);
  box-shadow: 0 0 0 3px var(--p10);
}
#rcw-input::placeholder { color: var(--text3); }
#rcw-input:disabled { opacity: .5; }
#rcw-send {
  width: 40px; height: 40px; border-radius: 12px;
  background: var(--p);
  border: none; color: #fff; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; align-self: flex-end;
  transition: transform .2s, background .2s, opacity .2s, box-shadow .2s;
  box-shadow: 0 2px 10px var(--p30);
}
#rcw-send:hover:not(:disabled) {
  transform: scale(1.08); background: var(--pdk);
  box-shadow: 0 4px 16px var(--p30);
}
#rcw-send:disabled { opacity: .4; cursor: not-allowed; transform: none; box-shadow: none; }
#rcw-send svg { width: 16px; height: 16px; }

/* ── Empty state ── */
.rcw-empty {
  flex: 1;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; padding: 32px;
  text-align: center;
}
.rcw-empty-icon {
  width: 48px; height: 48px; border-radius: 14px;
  background: var(--p);
  color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; margin-bottom: 4px;
}
.rcw-empty-title {
  font-size: 14px; font-weight: 600; color: var(--text);
}
.rcw-empty-sub {
  font-size: 12.5px; color: var(--text2); line-height: 1.6;
  max-width: 240px;
}

/* ── Mobile ── */
@media (max-width: 500px) {
  #rcw-window {
    width: 100vw; height: 100dvh; max-height: 100dvh;
    border-radius: 0;
  }
  #rcw-window.rcw-open { animation: rcwSlideUp .28s ease-out both; }
  @keyframes rcwSlideUp {
    from { opacity:0; transform: translateY(100%); }
    to   { opacity:1; transform: translateY(0); }
  }
}
    `;
    document.head.appendChild(el);
  }

  /* ═══════════════════════════════════════════
   * 5. SEND ICON SVG
   * ═══════════════════════════════════════════ */
  const SEND_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;

  /* ═══════════════════════════════════════════
   * 6. RENDER ENGINE
   * ═══════════════════════════════════════════
   *
   * Problem: marked.js breaks multiline $$...$$ 
   * into separate <p> tags, so renderMathInElement
   * can no longer find $$ delimiters.
   *
   * Solution:
   *   1. _protectMath() - extract $$...$$ & \[...\],
   *      replace with unique placeholder strings
   *   2. marked.parse() - marked ignores placeholders
   *   3. _restoreMath() - put original math back
   *   4. renderMathInElement() - sees clean $$ now
   * ═══════════════════════════════════════════ */

  /** Step 1 – pull block math out before marked touches it */
  function _protectMath(text) {
    const store = [];

    // Handle $$...$$ BEFORE \[...\] to avoid double-match
    const protected1 = text.replace(
      /\$\$([\s\S]*?)\$\$/g,
      (match) => {
        const idx = store.length;
        store.push(match);
        return `RCWMATH_BLOCK_${idx}_END`;
      }
    );

    const protected2 = protected1.replace(
      /\\\[([\s\S]*?)\\\]/g,
      (match) => {
        const idx = store.length;
        store.push(match);
        return `RCWMATH_BLOCK_${idx}_END`;
      }
    );

    return { text: protected2, store };
  }

  /** Step 3 – restore placeholders back into the rendered HTML */
  function _restoreMath(html, store) {
    return html.replace(/RCWMATH_BLOCK_(\d+)_END/g, (_, i) => {
      return store[parseInt(i, 10)] || '';
    });
  }

  function _renderMarkdown(raw) {
    if (!window.marked || !window.DOMPurify) {
      return `<span style="white-space:pre-wrap">${_esc(raw)}</span>`;
    }

    // Step 1: Protect block math before marked touches it
    const { text: safeText, store } = _protectMath(raw);

    const renderer = new marked.Renderer();

    renderer.code = (code, lang) => {
      const text = typeof code === 'object' ? code.text : (code || '');
      const language = typeof code === 'object' ? code.lang : (lang || 'text');
      const safeLang = _esc(language || 'text');
      const safeCode = _esc(text);
      const dataCode = safeCode.replace(/"/g, '&quot;');
      return `<div class="rcw-code-wrap">
  <div class="rcw-code-hdr">
    <span class="rcw-code-lang">${safeLang}</span>
    <button class="rcw-copy-btn" data-copycode="${dataCode}">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      Copy
    </button>
  </div>
  <pre><code class="language-${_esc(language || 'text')}">${safeCode}</code></pre>
</div>`;
    };

    renderer.codespan = (code) => {
      const text = typeof code === 'object' ? code.text : (code || '');
      return `<code>${_esc(text)}</code>`;
    };

    marked.use({ renderer, breaks: true, gfm: true });

    // Step 2: marked.parse on protected text (math is now placeholder)
    const rawHtml = marked.parse(safeText);

    // Step 3: Restore math strings back into HTML
    const restoredHtml = _restoreMath(rawHtml, store);

    // Step 4: Sanitize
    return DOMPurify.sanitize(restoredHtml, {
      ADD_TAGS: ['div', 'span', 'button'],
      ADD_ATTR: ['class', 'data-copycode', 'style'],
      FORCE_BODY: false,
    });
  }

  function _highlightCode(container) {
    if (!window.hljs) return;
    container.querySelectorAll('pre code').forEach(block => {
      try { hljs.highlightElement(block); } catch (_) { /* ignore */ }
    });
  }

  function _renderKaTeX(container) {
    if (!window.renderMathInElement) return;
    try {
      renderMathInElement(container, {
        delimiters: [
          { left: '$$',  right: '$$',  display: true  },
          { left: '$',   right: '$',   display: false },
          { left: '\\[', right: '\\]', display: true  },
          { left: '\\(', right: '\\)', display: false },
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
        throwOnError: false,
        strict: false,
        trust: false,
      });
    } catch (_) { /* ignore */ }
  }

  function _attachCopyBtns(container) {
    container.querySelectorAll('.rcw-copy-btn').forEach(btn => {
      if (btn._rcwBound) return;
      btn._rcwBound = true;
      btn.addEventListener('click', async () => {
        const raw = btn.dataset.copycode || '';
        const text = raw
          .replace(/&quot;/g, '"')
          .replace(/&amp;/g, '&')
          .replace(/&lt;/g, '<')
          .replace(/&gt;/g, '>');
        try {
          await navigator.clipboard.writeText(text);
          btn.classList.add('rcw-copied');
          btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
          setTimeout(() => {
            btn.classList.remove('rcw-copied');
            btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
          }, 2000);
        } catch (_) { /* clipboard unavailable */ }
      });
    });
  }

  function _parseSources(text) {
    /* Detect the 📚 **Nguồn tham khảo:** block and return { before, sources[] } */
    const markerIdx = text.indexOf('📚');
    if (markerIdx === -1) return { before: text, sources: [] };

    const beforeText = text.slice(0, markerIdx).trim();
    const srcBlock   = text.slice(markerIdx);
    const lines      = srcBlock.split('\n').filter(Boolean);
    const sources    = lines.slice(1)
      .map(l => l.replace(/^\s*[-*•]\s*/, '').trim())
      .filter(Boolean);

    return { before: beforeText, sources };
  }

  function _buildSourcesHTML(sources) {
    if (!sources.length) return '';
    return `<div class="rcw-sources">
  <div class="rcw-sources-hdr">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
    Nguồn tham khảo
  </div>
  ${sources.map(s => `<div class="rcw-source-item"><div class="rcw-source-dot"></div><span>${_esc(s)}</span></div>`).join('')}
</div>`;
  }

  function _fullRender(bubble, raw) {
    const { before, sources } = _parseSources(raw);
    bubble.innerHTML = _renderMarkdown(before) + _buildSourcesHTML(sources);
    _highlightCode(bubble);
    _renderKaTeX(bubble);
    _attachCopyBtns(bubble);
  }

  /* ═══════════════════════════════════════════
   * 7. CHAT WIDGET CLASS
   * ═══════════════════════════════════════════ */
  class ChatWidget {
    constructor(cfg = {}) {
      this.cfg      = { ...DEFAULTS, ...cfg };
      this.isOpen   = false;
      this.dark     = this._detectDark();
      this.subject  = this.cfg.subject || null;
      this.subjects = [];
      this._abort   = null;
      this._libsReady = false;
      this._msgCount  = 0;  // for grouping

      _injectCSS(this.cfg.primaryColor, this.cfg.accentColor);
      this._build();
      this._bind();
      this._fetchSubjects();
      _loadAllLibs().then(() => { this._libsReady = true; });
    }

    _detectDark() {
      if (this.cfg.theme === 'dark')  return true;
      if (this.cfg.theme === 'light') return false;
      return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
    }

    /* ── Position helpers ── */
    _pos() {
      const p = this.cfg.position;
      const map = {
        'bottom-right': { pos: 'bottom:20px;right:20px', win: 'bottom:70px;right:20px', orig: 'bottom right' },
        'bottom-left':  { pos: 'bottom:20px;left:20px',  win: 'bottom:70px;left:20px',  orig: 'bottom left'  },
        'top-right':    { pos: 'top:20px;right:20px',    win: 'top:70px;right:20px',    orig: 'top right'    },
        'top-left':     { pos: 'top:20px;left:20px',     win: 'top:70px;left:20px',     orig: 'top left'     },
      };
      return map[p] || map['bottom-right'];
    }

    /* ── Build DOM ── */
    _build() {
      this._root = document.createElement('div');
      this._root.id = 'rcw-v3-root';
      if (this.dark) this._root.classList.add('dark');

      const pos = this._pos();

      this._root.innerHTML = `
        <button id="rcw-launcher" style="${pos.pos}" title="Chat với AI">${_esc(this.cfg.launchEmoji)}<span id="rcw-badge"></span></button>
        <div id="rcw-window" style="${pos.win};transform-origin:${pos.orig}">
          <div id="rcw-header">
            <div class="rcw-hdr-avatar">${_esc(this.cfg.botEmoji)}</div>
            <div class="rcw-hdr-info">
              <div class="rcw-hdr-title">${_esc(this.cfg.title)}</div>
              <div class="rcw-hdr-sub">
                <div class="rcw-status-dot"></div>
                <span class="rcw-hdr-sub-txt" id="rcw-status">Online</span>
              </div>
            </div>
            <div class="rcw-hdr-actions">
              <button class="rcw-hbtn" id="rcw-theme-btn" title="Toggle dark mode">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
              </button>
              <button class="rcw-hbtn" id="rcw-clear-btn" title="Xoá chat">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
              </button>
              <button class="rcw-hbtn" id="rcw-close-btn" title="Đóng">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>
          </div>

          <div id="rcw-subj-bar">
            <span class="rcw-subj-label">Môn học</span>
            <div class="rcw-chip neutral" id="rcw-chip">
              <span class="rcw-chip-name" id="rcw-chip-name">Tất cả</span>
              <span class="rcw-chip-x" id="rcw-chip-x" style="display:none">×</span>
            </div>
            <div id="rcw-subj-popover">
              <div class="rcw-pop-search">
                <input id="rcw-pop-input" type="text" placeholder="Tìm môn học…" autocomplete="off"/>
              </div>
              <div class="rcw-pop-list" id="rcw-pop-list">
                <div class="rcw-pop-empty">Đang tải…</div>
              </div>
            </div>
          </div>

          <div id="rcw-msgs">
            <div class="rcw-divider">Hôm nay</div>
          </div>

          <div id="rcw-typing">
            <div class="rcw-typing-bubble">
              <div class="rcw-tdot"></div>
              <div class="rcw-tdot"></div>
              <div class="rcw-tdot"></div>
            </div>
          </div>

          <div id="rcw-input-area">
            <textarea id="rcw-input" rows="1" placeholder="${_esc(this.cfg.placeholder)}"></textarea>
            <button id="rcw-send">${SEND_SVG}</button>
          </div>
        </div>
      `;

      // Position root element itself so launcher and window are absolutely placed inside
      this._root.style.cssText = 'position:fixed;z-index:2147483640;';
      document.body.appendChild(this._root);

      this.$launcher   = this._root.querySelector('#rcw-launcher');
      this.$window     = this._root.querySelector('#rcw-window');
      this.$msgs       = this._root.querySelector('#rcw-msgs');
      this.$typing     = this._root.querySelector('#rcw-typing');
      this.$input      = this._root.querySelector('#rcw-input');
      this.$send       = this._root.querySelector('#rcw-send');
      this.$chip       = this._root.querySelector('#rcw-chip');
      this.$chipName   = this._root.querySelector('#rcw-chip-name');
      this.$chipX      = this._root.querySelector('#rcw-chip-x');
      this.$popover    = this._root.querySelector('#rcw-subj-popover');
      this.$popList    = this._root.querySelector('#rcw-pop-list');
      this.$popInput   = this._root.querySelector('#rcw-pop-input');
      this.$status     = this._root.querySelector('#rcw-status');

      // Append welcome
      this._appendBot(this.cfg.welcomeMessage);
    }

    /* ── Events ── */
    _bind() {
      this.$launcher.addEventListener('click', () => this._toggle());
      this._root.querySelector('#rcw-close-btn').addEventListener('click', () => this._close());
      this._root.querySelector('#rcw-theme-btn').addEventListener('click', () => this._toggleDark());
      this._root.querySelector('#rcw-clear-btn').addEventListener('click', () => this._clear());

      this.$send.addEventListener('click', () => this._send());
      this.$input.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this._send(); }
      });
      this.$input.addEventListener('input', () => this._resizeInput());

      this.$chip.addEventListener('click', e => {
        if (e.target === this.$chipX) return;
        this._togglePopover();
      });
      this.$chipX.addEventListener('click', e => { e.stopPropagation(); this._selectSubject(null); });
      this.$popInput.addEventListener('input', () => this._filterPop(this.$popInput.value));

      document.addEventListener('click', e => {
        if (!this._root.querySelector('#rcw-subj-bar').contains(e.target)) {
          this._closePopover();
        }
      });

      if (this.cfg.theme === 'auto') {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
          this.dark = e.matches;
          this._root.classList.toggle('dark', this.dark);
        });
      }
    }

    /* ── Window ── */
    _toggle() { this.isOpen ? this._close() : this._open(); }
    _open() {
      this.isOpen = true;
      this.$launcher.classList.add('rcw-hidden');
      this.$window.classList.add('rcw-open');
      this._root.querySelector('#rcw-badge').style.display = 'none';
      setTimeout(() => this.$input.focus(), 150);
      this._scrollBottom();
    }
    _close() {
      this.isOpen = false;
      this.$launcher.classList.remove('rcw-hidden');
      this.$window.classList.remove('rcw-open');
      this._closePopover();
    }
    _toggleDark() {
      this.dark = !this.dark;
      this._root.classList.toggle('dark', this.dark);
    }
    _clear() {
      this.$msgs.innerHTML = '<div class="rcw-divider">Hôm nay</div>';
      this._msgCount = 0;
      this._appendBot(this.cfg.welcomeMessage);
    }

    /* ── Subject ── */
    async _fetchSubjects() {
      try {
        const r = await fetch(`${this.cfg.apiUrl}/subjects`);
        const d = await r.json();
        this.subjects = d.subjects || [];
      } catch (_) {
        this.subjects = this.cfg.subjects || [];
      }
      this._renderPopList(this.subjects);
    }

    _renderPopList(list) {
      this.$popList.innerHTML = '';

      const allEl = document.createElement('div');
      allEl.className = 'rcw-pop-item' + (this.subject === null ? ' active' : '');
      allEl.innerHTML = `<span>📚</span><span>Tất cả môn học</span>`;
      allEl.addEventListener('click', () => this._selectSubject(null));
      this.$popList.appendChild(allEl);

      if (!list.length) {
        const em = document.createElement('div');
        em.className = 'rcw-pop-empty'; em.textContent = 'Không có môn học';
        this.$popList.appendChild(em);
        return;
      }

      const emoji = n => {
        const s = (n || '').toLowerCase();
        if (/toán|math/.test(s))       return '📐';
        if (/vật lý|physics/.test(s))  return '⚡';
        if (/hóa|chem/.test(s))        return '🧪';
        if (/sinh|bio/.test(s))        return '🧬';
        if (/lập trình|code|prog/.test(s)) return '💻';
        if (/mạng|network/.test(s))    return '🌐';
        if (/cơ sở dữ liệu|database/.test(s)) return '🗄️';
        if (/trí tuệ|ai|ml/.test(s))   return '🤖';
        if (/kinh tế|econ/.test(s))    return '📈';
        if (/văn|lit/.test(s))         return '📖';
        return '📘';
      };

      list.forEach(s => {
        const el = document.createElement('div');
        el.className = 'rcw-pop-item' + (this.subject === s ? ' active' : '');
        el.dataset.subj = s;
        el.innerHTML = `<span>${emoji(s)}</span><span>${_esc(s)}</span>`;
        el.addEventListener('click', () => this._selectSubject(s));
        this.$popList.appendChild(el);
      });
    }

    _filterPop(q) {
      const lq = q.toLowerCase();
      this.$popList.querySelectorAll('.rcw-pop-item').forEach(el => {
        const name = (el.dataset.subj || '').toLowerCase();
        el.style.display = (!lq || !el.dataset.subj || name.includes(lq)) ? '' : 'none';
      });
    }

    _selectSubject(s) {
      this.subject = s;
      this.$chipName.textContent = s || 'Tất cả';
      this.$chipX.style.display = s ? '' : 'none';
      this.$chip.className = 'rcw-chip ' + (s ? 'active' : 'neutral');
      this.$popList.querySelectorAll('.rcw-pop-item').forEach(el => {
        const isAll = !el.dataset.subj;
        el.classList.toggle('active', s === null ? isAll : el.dataset.subj === s);
      });
      this._closePopover();
    }

    _togglePopover() {
      const open = this.$popover.classList.toggle('rcw-open');
      if (open) {
        this.$popInput.value = '';
        this._filterPop('');
        this.$popInput.focus();
      }
    }
    _closePopover() { this.$popover.classList.remove('rcw-open'); }

    /* ── Messaging ── */
    _appendUser(text) {
      this._msgCount++;
      const row = document.createElement('div');
      row.className = 'rcw-row rcw-user rcw-group-start';
      row.innerHTML = `
        <div>
          <div class="rcw-bubble">${_esc(text).replace(/\n/g, '<br>')}</div>
          <div class="rcw-meta">${_now()}</div>
        </div>`;
      this.$msgs.appendChild(row);
      this._scrollBottom();
    }

    _appendBot(text, render = false) {
      this._msgCount++;
      const row = document.createElement('div');
      row.className = 'rcw-row rcw-bot rcw-group-start';
      const bubble = document.createElement('div');
      bubble.className = 'rcw-bubble';

      if (render && this._libsReady) {
        _fullRender(bubble, text);
      } else if (render) {
        // will render once libs ready
        bubble.dataset.pendingRender = text;
        bubble.textContent = text;
      } else {
        bubble.innerHTML = _esc(text);
      }

      const meta = document.createElement('div');
      meta.className = 'rcw-meta';
      meta.textContent = _now();

      const wrap = document.createElement('div');
      wrap.appendChild(bubble);
      wrap.appendChild(meta);
      row.appendChild(wrap);
      this.$msgs.appendChild(row);
      this._scrollBottom();
      return bubble;
    }

    /* Streaming */
    _appendBotStreaming() {
      this._msgCount++;
      const row = document.createElement('div');
      row.className = 'rcw-row rcw-bot rcw-group-start';
      const bubble = document.createElement('div');
      bubble.className = 'rcw-bubble';
      const pre = document.createElement('span');
      pre.style.whiteSpace = 'pre-wrap';
      bubble.appendChild(pre);

      const meta = document.createElement('div');
      meta.className = 'rcw-meta';
      meta.textContent = _now();

      const wrap = document.createElement('div');
      wrap.appendChild(bubble);
      wrap.appendChild(meta);
      row.appendChild(wrap);
      this.$msgs.appendChild(row);
      return { bubble, streamSpan: pre };
    }

    _finaliseStreaming(bubble, raw) {
      _fullRender(bubble, raw);
      this._scrollBottom();
    }

    /* ── Send ── */
    async _send() {
      const text = this.$input.value.trim();
      if (!text) return;

      this.$input.value = '';
      this._resizeInput();
      this._setEnabled(false);
      this._appendUser(text);
      this._showTyping(true);
      this._setStatus('Đang trả lời…');

      if (this._abort) this._abort.abort();
      this._abort = new AbortController();

      const { bubble, streamSpan } = this._appendBotStreaming();
      let raw = '';

      try {
        const res = await fetch(`${this.cfg.apiUrl}/query/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: this._abort.signal,
          body: JSON.stringify({
            question:    text,
            subject:     this.subject,
            model_key:   this.cfg.modelKey,
            use_history: true,
          }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this._showTyping(false);

        const reader = res.body.getReader();
        const dec    = new TextDecoder();
        let buf = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const parts = buf.split('\n\n');
          buf = parts.pop() ?? '';
          for (const part of parts) {
            if (!part.startsWith('data: ')) continue;
            try {
              const d = JSON.parse(part.slice(6));
              if (d.error) throw new Error(d.error);
              if (d.done)  break;
              if (d.chunk) {
                raw += d.chunk;
                streamSpan.textContent = raw;
                this._scrollBottom();
              }
            } catch (_) { /* non-JSON chunk */ }
          }
        }

        this._finaliseStreaming(bubble, raw);

      } catch (err) {
        this._showTyping(false);
        if (err.name !== 'AbortError') {
          bubble.innerHTML = `<span style="color:#f87171">❌ Lỗi: ${_esc(err.message)}</span>`;
        }
      }

      this._setEnabled(true);
      this._setStatus('Online');
      this.$input.focus();
    }

    /* ── UI helpers ── */
    _showTyping(show) {
      this.$typing.classList.toggle('rcw-vis', show);
      if (show) this._scrollBottom();
    }
    _setEnabled(enabled) {
      this.$input.disabled = !enabled;
      this.$send.disabled  = !enabled;
    }
    _setStatus(t) { this.$status.textContent = t; }
    _scrollBottom() {
      requestAnimationFrame(() => {
        this.$msgs.scrollTop = this.$msgs.scrollHeight;
      });
    }
    _resizeInput() {
      this.$input.style.height = 'auto';
      this.$input.style.height = Math.min(this.$input.scrollHeight, 130) + 'px';
    }
  }

  /* ═══════════════════════════════════════════
   * 8. AUTO-INIT FROM SCRIPT ATTRIBUTES
   * ═══════════════════════════════════════════ */
  function _autoInit() {
    const s = document.querySelector('script[data-widget-init], script[data-api-url]');
    if (!s) return;
    const g = k => s.getAttribute(k) || undefined;
    const cfg = {
      apiUrl:         g('data-api-url'),
      subject:        g('data-subject')         || null,
      modelKey:       g('data-model-key')        || null,
      position:       g('data-position'),
      primaryColor:   g('data-primary-color'),
      accentColor:    g('data-accent-color'),
      title:          g('data-title'),
      subtitle:       g('data-subtitle'),
      botEmoji:       g('data-bot-emoji'),
      launchEmoji:    g('data-launch-emoji'),
      welcomeMessage: g('data-welcome-message'),
      placeholder:    g('data-placeholder'),
      theme:          g('data-theme'),
    };
    // Remove undefineds
    Object.keys(cfg).forEach(k => cfg[k] === undefined && delete cfg[k]);
    window.rcwWidget = new ChatWidget(cfg);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _autoInit);
  } else {
    _autoInit();
  }

  // Export
  window.ChatWidget = ChatWidget;
})();