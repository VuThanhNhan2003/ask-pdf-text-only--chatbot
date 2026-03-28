"""
Frontend styles — VNUHCM-ITP Smart Assistant
Safe CSS injection for Streamlit 1.49.
"""
import streamlit as st
import base64
from pathlib import Path

LOGO_DIR = Path(__file__).parent / "logo_image"


def load_image_base64(filename: str) -> str:
    """Load logo file as base64 data-URI. Tries png/jpg/jpeg/svg/webp."""
    for ext in ["png", "jpg", "jpeg", "svg", "webp"]:
        path = LOGO_DIR / f"{filename}.{ext}"
        if path.exists():
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    return ""


def inject_global_css():
    """Inject all global CSS. Call once at app startup before any render."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --navy:   #0d2b4e;
    --blue:   #1a4a7a;
    --accent: #2563eb;
    --acc2:   #1d4ed8;
    --bg:     #f8fafc;
    --bg2:    #ffffff;
    --bdr:    #e2e8f0;
    --txt:    #0f172a;
    --txt2:   #475569;
    --txt3:   #94a3b8;
    --sans:   'Be Vietnam Pro', system-ui, sans-serif;
}

/* ── Base font ── */
html, body, .stApp { font-family: var(--sans) !important; }

/* ── Hide only deploy button, keep native header/toolbar intact ── */
.stDeployButton { display: none !important; }

/* Tighten the native header height slightly */
header[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* ── Main content padding ── */
.main .block-container {
    padding-top: 0 !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--navy) !important; }
[data-testid="stSidebar"] > div:first-child {
    background: var(--navy) !important;
    padding-top: 1rem !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small { color: rgba(255,255,255,0.85) !important; }

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 0.6rem 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.38) !important;
    margin-bottom: 5px !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    background: transparent !important;
    color: white !important;
    border: none !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: rgba(255,255,255,0.5) !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    border: none !important;
    text-align: left !important;
    transition: background 0.15s !important;
    font-family: var(--sans) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: var(--acc2) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.13) !important;
    color: white !important;
}
[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
    background: transparent !important;
    color: rgba(255,255,255,0.7) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
}

/* Expander in sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: rgba(255,255,255,0.6) !important;
    font-size: 12px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 3px 0 !important;
}
/* User: flip to right side */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
  [data-testid="stMarkdownContainer"] {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 16px !important;
    max-width: 72% !important;
    margin-left: auto !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
    color: white !important;
}
/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
  [data-testid="stMarkdownContainer"] {
    background: white !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 10px 16px !important;
    max-width: 80% !important;
    border: 1px solid var(--bdr) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatMessage"] p {
    font-size: 14px !important;
    line-height: 1.72 !important;
    margin-bottom: 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) code {
    background: #f1f5f9 !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    font-size: 12.5px !important;
}

/* ── Chat input ── */
[data-testid="stBottom"] {
    background: var(--bg2) !important;
    border-top: 1px solid var(--bdr) !important;
    padding: 10px 16px 14px !important;
}
[data-testid="stChatInput"] > div {
    border-radius: 14px !important;
    border: 1.5px solid var(--bdr) !important;
    background: var(--bg) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    background: white !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 14px !important;
    color: var(--txt) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--txt3) !important; }

/* ── Misc ── */
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 13px !important; }
[data-testid="stExpander"] { border-radius: 10px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); }

/* ── Typing dots (used in chat_ui) ── */
@keyframes blink {
    0%, 100% { opacity: 0.25; transform: scale(0.85); }
    50%       { opacity: 1;   transform: scale(1.1);  }
}
.tdot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #94a3b8;
    animation: blink 1.3s ease-in-out infinite;
}
.tdot:nth-child(2) { animation-delay: 0.2s; }
.tdot:nth-child(3) { animation-delay: 0.4s; }

/* ── Subject badge ── */
.subject-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


def render_app_header():
    """Sticky top header: logos left · title center · user avatar right."""
    logo_itp    = load_image_base64("logo_itp")
    logo_vnuhcm = load_image_base64("logo_vnuhcm")

    itp_html = (
        f'<img src="{logo_itp}" height="34" alt="ITP" style="display:block"/>'
        if logo_itp else
        '<b style="font-size:14px;color:#0d2b4e;letter-spacing:0.05em">ITP</b>'
    )
    vnuhcm_html = (
        f'<img src="{logo_vnuhcm}" height="34" alt="VNUHCM" style="display:block"/>'
        if logo_vnuhcm else
        '<b style="font-size:14px;color:#0d2b4e;letter-spacing:0.05em">VNUHCM</b>'
    )

    user     = st.session_state.get("user") or {}
    username = user.get("username", "")
    initials = username[:2].upper() if username else "U"
    user_chip = (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="width:30px;height:30px;border-radius:50%;'
        f'background:linear-gradient(135deg,#1a4a7a,#2563eb);'
        f'color:white;font-size:11px;font-weight:600;'
        f'display:flex;align-items:center;justify-content:center;">{initials}</div>'
        f'<span style="font-size:13px;font-weight:500;color:#475569;">{username}</span>'
        f'</div>'
    ) if username else ""

    st.markdown(f"""
<div style="
    display:flex; align-items:center; justify-content:space-between;
    padding:0 24px; height:56px; background:#fff;
    border-bottom:1px solid #e2e8f0; margin-bottom:0;
    position:relative; box-shadow:0 1px 3px rgba(0,0,0,0.06);
    font-family:'Be Vietnam Pro',sans-serif;
">
  <div style="display:flex;align-items:center;gap:14px;">
    {vnuhcm_html}
    <div style="width:1px;height:26px;background:#e2e8f0;"></div>
    {itp_html}
  </div>
  <div style="
    position:absolute; left:50%; transform:translateX(-50%);
    font-size:14px; font-weight:600; color:#0f172a;
    letter-spacing:-0.01em; white-space:nowrap;
  ">VNUHCM-ITP Smart Assistant</div>
  {user_chip}
</div>
""", unsafe_allow_html=True)


def render_sidebar_header():
    """Brand block at top of the dark sidebar."""
    logo_chatbot = load_image_base64("logo_chatbot")
    avatar_html = (
        f'<img src="{logo_chatbot}" width="38" height="38"'
        f' style="border-radius:10px;flex-shrink:0;object-fit:cover;" alt=""/>'
        if logo_chatbot else
        '<div style="width:38px;height:38px;border-radius:10px;'
        'background:linear-gradient(135deg,#1a4a7a,#2563eb);'
        'display:flex;align-items:center;justify-content:center;font-size:18px;">🎓</div>'
    )
    st.markdown(f"""
<div style="
    display:flex; align-items:center; gap:10px;
    padding-bottom:14px; margin-bottom:10px;
    border-bottom:1px solid rgba(255,255,255,0.1);
    font-family:'Be Vietnam Pro',sans-serif;
">
  {avatar_html}
  <div>
    <div style="font-size:14px;font-weight:600;color:white;line-height:1.3;">
      Smart Assistant</div>
    <div style="font-size:10px;color:rgba(255,255,255,0.4);
        letter-spacing:0.08em;text-transform:uppercase;margin-top:1px;">
      VNUHCM · ITP</div>
  </div>
</div>
""", unsafe_allow_html=True)