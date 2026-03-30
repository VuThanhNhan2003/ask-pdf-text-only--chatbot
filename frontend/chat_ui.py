"""
Chat UI helpers — VNUHCM-ITP Smart Assistant
Uses native st.chat_message() + st.markdown(). No HTML injected inside chat containers.
CSS bubble styling is handled entirely by styles.py via data-testid selectors.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64


LOGO_DIR = Path(__file__).parent / "logo_image"


def _load_b64(filename: str) -> str:
    """Load image as base64 data-URI."""
    for ext in ["png", "jpg", "jpeg", "svg", "webp"]:
        p = LOGO_DIR / f"{filename}.{ext}"
        if p.exists():
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    return ""


def get_avatar_uri() -> str:
    """Cache chatbot avatar in session_state so we only load it once."""
    if "_bot_avatar" not in st.session_state:
        st.session_state._bot_avatar = _load_b64("logo_chatbot")
    return st.session_state._bot_avatar


# ─── Single message ──────────────────────────────────────────────────────────

def render_message(role: str, content: str):
    """
    Render one message with st.chat_message() + st.markdown().
    This is the correct Streamlit pattern — no custom HTML inside the container.
    Bubble styling (background, border-radius, alignment) comes from styles.py CSS.
    """
    avatar = get_avatar_uri()

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        # Pass logo image as avatar if available, else fall back to "assistant" icon
        with st.chat_message("assistant", avatar=avatar if avatar else "assistant"):
            st.markdown(content)


def render_messages(messages: list):
    """Render a full list of message dicts [{role, content}, ...]."""
    for msg in messages:
        render_message(msg["role"], msg["content"])


# ─── Streaming response ───────────────────────────────────────────────────────

def render_streaming_response(processor, user_input: str) -> str:
    """
    Stream LLM response inside a proper st.chat_message() container.
    Shows text token-by-token with a cursor, then finalises without cursor.
    Returns the full response string.
    """
    avatar = get_avatar_uri()
    full_response = ""

    with st.chat_message("assistant", avatar=avatar if avatar else "assistant"):
        placeholder = st.empty()
        for chunk in processor.get_response_stream(user_input, use_history=True):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        # Final render without blinking cursor
        placeholder.markdown(full_response)

    return full_response


# ─── Typing indicator ────────────────────────────────────────────────────────

def render_typing_indicator():
    """
    Show animated 3-dot 'thinking' indicator while waiting for the first chunk.
    Returns the st.empty() container — call .empty() to dismiss it.
    """
    avatar = get_avatar_uri()
    avatar_style = (
        f"background:url('{avatar}') center/cover no-repeat;"
        if avatar else
        "background:linear-gradient(135deg,#1a4a7a,#2563eb);"
        "display:flex;align-items:center;justify-content:center;"
        "font-size:11px;font-weight:700;color:white;"
    )
    avatar_inner = "" if avatar else "AI"

    placeholder = st.empty()
    placeholder.markdown(f"""
<div style="display:flex;align-items:flex-end;gap:10px;padding:4px 0;">
  <div style="width:30px;height:30px;border-radius:8px;flex-shrink:0;{avatar_style}">
    {avatar_inner}
  </div>
  <div style="
    display:inline-flex;align-items:center;gap:5px;
    background:white;border:1px solid #e2e8f0;
    border-radius:18px 18px 18px 4px;
    padding:12px 16px;
    box-shadow:0 1px 4px rgba(0,0,0,0.05);
  ">
    <span class="tdot"></span>
    <span class="tdot"></span>
    <span class="tdot"></span>
  </div>
</div>
""", unsafe_allow_html=True)
    return placeholder


# ─── Empty / welcome state ────────────────────────────────────────────────────

def render_empty_state():
    """Shown when no conversation is open yet."""
    avatar = get_avatar_uri()
    avatar_html = (
        f'<img src="{avatar}" width="56" height="56"'
        f' style="border-radius:14px;object-fit:cover;" alt=""/>'
        if avatar else
        '<div style="width:56px;height:56px;border-radius:14px;'
        'background:linear-gradient(135deg,#0d2b4e,#2563eb);"></div>'
    )
    st.markdown(f"""
<div style="
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; padding:80px 24px; text-align:center;
    font-family:'Be Vietnam Pro',sans-serif;
">
  <div style="margin-bottom:20px;filter:drop-shadow(0 6px 16px rgba(37,99,235,0.2));">
    {avatar_html}
  </div>
  <div style="font-size:18px;font-weight:600;color:#0f172a;margin-bottom:8px;">
    Xin chào! Tôi là Smart Assistant</div>
  <div style="font-size:14px;color:#64748b;max-width:380px;line-height:1.7;">
    Chọn một cuộc trò chuyện bên trái hoặc nhấn
    <strong>+ New Chat</strong> để bắt đầu.<br/>
    Tôi có thể giúp bạn tìm hiểu tài liệu môn học nhanh chóng và chính xác.
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Subject badge ────────────────────────────────────────────────────────────

def render_subject_badge(subject: str | None):
    """Small pill at top of chat showing active subject filter."""
    label = subject if subject else "Tất cả môn học"
    st.markdown(
        f'<div class="subject-badge">📖 {label}</div>',
        unsafe_allow_html=True,
    )


# ─── Auto-scroll ─────────────────────────────────────────────────────────────

def inject_auto_scroll():
    """Scroll the chat window to the bottom after a new message. Renders invisibly (height=0)."""
    components.html("""
<script>
(function(){
    function scroll(){
        try {
            var p = window.parent;
            [
                p.document.querySelector('[data-testid="stMain"]'),
                p.document.querySelector('.main'),
                p.document.documentElement,
                p.document.body,
            ].forEach(function(el){ if(el) el.scrollTop = el.scrollHeight; });
        } catch(e){}
        window.scrollTo(0, document.body.scrollHeight);
    }
    scroll();
    setTimeout(scroll, 150);
    setTimeout(scroll, 400);
})();
</script>
""", height=0, scrolling=False)