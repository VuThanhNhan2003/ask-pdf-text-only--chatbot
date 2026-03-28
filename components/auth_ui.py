"""
Streamlit authentication UI components
"""
import streamlit as st
from auth.authentication import AuthService, AuthenticationError
from database.database import get_db
from frontend.streamlit_ui import render_auth_brand_header


def render_auth_brand():
    """Render app branding on auth page."""
    render_auth_brand_header()


def init_auth_state():
    """Initialize authentication session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None


def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.user_id = None
    st.session_state.chat_history = {}
    st.session_state.current_conversation_id = None
    st.rerun()


def render_login_form():
    """Render login form"""
    st.title("🔐 Đăng nhập")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Đăng nhập", use_container_width=True)
        
        if submit:
            if not email or not password:
                st.error("Vui lòng nhập đầy đủ thông tin")
                return
            
            try:
                with get_db() as db:
                    user = AuthService.authenticate_user(db, email, password)
                    
                    # Set session state
                    st.session_state.authenticated = True
                    st.session_state.user = {
                        'id': user.id,
                        'email': user.email,
                        'username': user.username,
                        'full_name': user.full_name
                    }
                    st.session_state.user_id = user.id
                    
                    st.success(f"Chào mừng {user.username}!")
                    st.rerun()
                    
            except AuthenticationError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Lỗi hệ thống: {str(e)}")


def render_signup_form():
    """Render signup form"""
    st.title("📝 Đăng ký tài khoản")
    
    with st.form("signup_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        username = st.text_input("Username", placeholder="username")
        full_name = st.text_input("Họ và tên (tùy chọn)", placeholder="Nguyễn Văn A")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Xác nhận password", type="password")
        
        submit = st.form_submit_button("Đăng ký", use_container_width=True)
        
        if submit:
            # Validation
            if not all([email, username, password, password_confirm]):
                st.error("Vui lòng nhập đầy đủ thông tin")
                return
            
            if password != password_confirm:
                st.error("Password không khớp")
                return
            
            try:
                with get_db() as db:
                    user = AuthService.create_user(
                        db=db,
                        email=email,
                        username=username,
                        password=password,
                        full_name=full_name if full_name else None
                    )
                    
                    st.success("✅ Đăng ký thành công! Vui lòng đăng nhập.")
                    st.balloons()
                    
            except AuthenticationError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Lỗi hệ thống: {str(e)}")


def render_auth_page():
    """Render main authentication page"""
    render_auth_brand()
    
    # Tabs for login/signup
    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])
    
    with tab1:
        render_login_form()
    
    with tab2:
        render_signup_form()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 RAG Chatbot với Streamlit + Qdrant + Gemini"
        "</div>",
        unsafe_allow_html=True
    )


def check_authentication():
    """
    Check if user is authenticated
    Call this at the start of protected pages
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    init_auth_state()
    
    if not st.session_state.authenticated:
        render_auth_page()
        st.stop()
        return False
    
    return True


def render_user_menu():
    """Render user menu in sidebar"""
    if st.session_state.user:
        user = st.session_state.user
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(
                "<div style='font-size:0.82rem; text-transform:uppercase; letter-spacing:.08em; color:#60708a; font-weight:700; margin-bottom:8px;'>Tài khoản</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**👤 {user['full_name'] or user['username']}**")
            st.caption(f"@{user['username']} • {user['email']}")
            
            if st.button("Đăng xuất", use_container_width=True):
                logout()