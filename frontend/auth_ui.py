"""
Auth UI — VNUHCM-ITP Smart Assistant
Centered card layout. No CSS positioning hacks, no broken column splits.
ALL auth logic (AuthService, session_state) is UNCHANGED from original.
"""
import streamlit as st
import base64
from pathlib import Path
from auth.authentication import AuthService, AuthenticationError
from database.database import get_db

LOGO_DIR = Path(__file__).parent.parent / "frontend" / "logo_image"


def _load_b64(filename: str) -> str:
    for ext in ["png", "jpg", "jpeg", "svg", "webp"]:
        p = LOGO_DIR / f"{filename}.{ext}"
        if p.exists():
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    return ""


def _inject_auth_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap');

html, body, .stApp { font-family: 'Be Vietnam Pro', system-ui, sans-serif !important; }

/* Deploy button only */
.stDeployButton { display: none !important; }

/* Light background for auth page */
.stApp { background: #f0f4fa !important; }

/* Center and constrain the form column */
.main .block-container {
    max-width: 480px !important;
    padding-top: 2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    margin: 0 auto !important;
}

/* ── Input fields ── */
.stTextInput > div > div > input {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 14px !important;
    border-radius: 9px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #f8fafc !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
    color: #0f172a !important;
}
.stTextInput > div > div > input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    background: white !important;
    outline: none !important;
}
.stTextInput label {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

/* ── Submit button ── */
.stForm .stButton > button {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    transition: background 0.2s, box-shadow 0.2s, transform 0.15s !important;
}
.stForm .stButton > button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}
.stForm .stButton > button:active { transform: translateY(0) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 8px 0 !important;
    flex: 1 !important;
    justify-content: center !important;
    border: none !important;
    background: transparent !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #0f172a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Card wrapper ── */
.auth-card {
    background: white;
    border-radius: 16px;
    padding: 32px 32px 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state — UNCHANGED ────────────────────────────────────────────────

def init_auth_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None


def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.user_id = None
    st.session_state.chat_history = {}
    st.session_state.current_conversation_id = None
    st.rerun()


# ─── Forms — logic UNCHANGED ─────────────────────────────────────────────────

def render_login_form():
    with st.form("login_form", clear_on_submit=False):
        email    = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Mật khẩu", type="password", placeholder="••••••••")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        submit   = st.form_submit_button("Đăng nhập →", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Vui lòng nhập đầy đủ thông tin")
                return
            try:
                with get_db() as db:
                    user = AuthService.authenticate_user(db, email, password)
                    st.session_state.authenticated = True
                    st.session_state.user = {
                        "id":        user.id,
                        "email":     user.email,
                        "username":  user.username,
                        "full_name": user.full_name,
                    }
                    st.session_state.user_id = user.id
                    st.success(f"Chào mừng {user.username}!")
                    st.rerun()
            except AuthenticationError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Lỗi hệ thống: {e}")


def render_signup_form():
    with st.form("signup_form", clear_on_submit=False):
        email    = st.text_input("Email", placeholder="your@email.com")
        username = st.text_input("Tên đăng nhập", placeholder="username")
        fullname = st.text_input("Họ và tên (tùy chọn)", placeholder="Nguyễn Văn A")
        pwd      = st.text_input("Mật khẩu", type="password", placeholder="••••••••")
        pwd2     = st.text_input("Xác nhận mật khẩu", type="password", placeholder="••••••••")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        submit   = st.form_submit_button("Tạo tài khoản →", use_container_width=True)

        if submit:
            if not all([email, username, pwd, pwd2]):
                st.error("Vui lòng nhập đầy đủ thông tin")
                return
            if pwd != pwd2:
                st.error("Mật khẩu không khớp")
                return
            try:
                with get_db() as db:
                    AuthService.create_user(
                        db=db, email=email, username=username,
                        password=pwd, full_name=fullname or None,
                    )
                    st.success("✅ Đăng ký thành công! Vui lòng đăng nhập.")
                    st.balloons()
            except AuthenticationError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Lỗi hệ thống: {e}")


# ─── Full auth page — centered card layout ────────────────────────────────────

def render_auth_page():
    """
    Centered single-column card layout.
    Uses layout="centered" + block_container max-width in CSS.
    No st.columns() split — that caused the broken left-panel code rendering.
    """
    st.set_page_config(
        page_title="VNUHCM-ITP Smart Assistant",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _inject_auth_css()

    logo_itp    = _load_b64("logo_itp")
    logo_vnuhcm = _load_b64("logo_vnuhcm")
    logo_bot    = _load_b64("logo_chatbot")

    itp_img = (
        f'<img src="{logo_itp}" height="32" alt="ITP" style="display:block"/>'
        if logo_itp else
        '<span style="font-weight:700;font-size:14px;color:#0d2b4e;letter-spacing:0.05em">ITP</span>'
    )
    vnuhcm_img = (
        f'<img src="{logo_vnuhcm}" height="32" alt="VNUHCM" style="display:block"/>'
        if logo_vnuhcm else
        '<span style="font-weight:700;font-size:14px;color:#0d2b4e;letter-spacing:0.05em">VNUHCM</span>'
    )
    bot_img = (
        f'<img src="{logo_bot}" width="52" height="52"'
        f' style="border-radius:14px;object-fit:cover;" alt=""/>'
        if logo_bot else
        '<div style="width:52px;height:52px;border-radius:14px;'
        'background:linear-gradient(135deg,#0d2b4e,#2563eb);'
        'display:flex;align-items:center;justify-content:center;font-size:26px;">🎓</div>'
    )

    # ── Header: dual org logos ──
    st.markdown(f"""
<div style="
    display:flex; align-items:center; justify-content:center;
    gap:16px; padding:24px 0 20px;
    font-family:'Be Vietnam Pro',sans-serif;
">
  {vnuhcm_img}
  <div style="width:1px;height:28px;background:#e2e8f0;"></div>
  {itp_img}
</div>
""", unsafe_allow_html=True)

    # ── Card: bot icon + title ──
    st.markdown(f"""
<div class="auth-card">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
    {bot_img}
    <div>
      <div style="font-family:'Be Vietnam Pro',sans-serif;
          font-size:18px;font-weight:600;color:#0f172a;line-height:1.2;">
        VNUHCM-ITP Smart Assistant</div>
      <div style="font-family:'Be Vietnam Pro',sans-serif;
          font-size:13px;color:#64748b;margin-top:3px;">
        Hệ thống hỗ trợ học tập thông minh</div>
    </div>
  </div>
  <div style="height:1px;background:#f1f5f9;margin-bottom:20px;"></div>
""", unsafe_allow_html=True)

    # Tabs inside the card
    tab1, tab2 = st.tabs(["🔑  Đăng nhập", "✏️  Đăng ký"])
    with tab1:
        render_login_form()
    with tab2:
        render_signup_form()

    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
<div style="
    text-align:center; padding:12px 0 32px;
    font-family:'Be Vietnam Pro',sans-serif;
    font-size:11px; color:#94a3b8; line-height:1.6;
">
  Đại học Quốc gia TP.HCM — Viện Công nghệ Thông tin<br/>
  © 2024 VNUHCM-ITP Smart Assistant
</div>
""", unsafe_allow_html=True)


# ─── Guards — UNCHANGED ───────────────────────────────────────────────────────

def check_authentication() -> bool:
    init_auth_state()
    if not st.session_state.authenticated:
        render_auth_page()
        st.stop()
        return False
    return True


# ─── Sidebar user menu ────────────────────────────────────────────────────────

def render_user_menu():
    """User avatar + name + logout, at bottom of the dark sidebar."""
    if not st.session_state.user:
        return

    user     = st.session_state.user
    username = user.get("username", "User")
    email    = user.get("email", "")
    initials = username[:2].upper()

    st.markdown(f"""
<div style="
    padding:10px 0 4px;
    border-top:1px solid rgba(255,255,255,0.1);
    font-family:'Be Vietnam Pro',sans-serif;
">
  <div style="display:flex;align-items:center;gap:10px;padding:6px 0 10px;">
    <div style="
        width:32px;height:32px;border-radius:8px;flex-shrink:0;
        background:linear-gradient(135deg,#1a4a7a,#2563eb);
        color:white;font-size:12px;font-weight:600;
        display:flex;align-items:center;justify-content:center;
    ">{initials}</div>
    <div style="min-width:0;">
      <div style="
          font-size:13px;font-weight:500;color:white;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
      ">{username}</div>
      <div style="
          font-size:11px;color:rgba(255,255,255,0.4);
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
      ">{email}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.button("↩ Đăng xuất", use_container_width=True, key="logout_btn"):
        logout()