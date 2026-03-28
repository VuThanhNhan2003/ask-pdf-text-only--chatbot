"""
VNUHCM-ITP Smart Assistant — Refactored UI layer
All backend logic (RAG, auth, database, services, session_state) is 100% unchanged.
Only the render layer has been replaced with frontend/ components.

Usage: rename this file to app_v2.py (or point your Streamlit run command at it).
"""
import streamlit as st
from processor import RAGProcessor, logger
import os
from config import config

# ── Backend imports — UNCHANGED ───────────────────────────────────────────────
from database.database import get_db, init_db
from services.conversation_service import ConversationService, MessageService

# ── Frontend layer — NEW ──────────────────────────────────────────────────────
from frontend.styles  import inject_global_css, render_app_header, render_sidebar_header
from frontend.chat_ui import (
    render_message,
    render_messages,
    render_streaming_response,
    render_empty_state,
    render_subject_badge,
    inject_auto_scroll,
    get_avatar_uri,
)
from frontend.auth_ui import check_authentication, render_user_menu, logout


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + INIT
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VNUHCM-ITP Smart Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize DB on first run — UNCHANGED
try:
    init_db()
except Exception:
    pass

# CSS must be injected before the auth gate (auth page also needs fonts/resets)
inject_global_css()

# Auth gate — shows login page if not authenticated, stops execution otherwise
check_authentication()


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE — UNCHANGED
# ═════════════════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "processor":               None,
        "current_conversation_id": None,
        "current_subject":         None,
        "current_model":           None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# DATA HELPERS — UNCHANGED logic
# ═════════════════════════════════════════════════════════════════════════════
def create_new_conversation(user_id: int, subject: str) -> int:
    with get_db() as db:
        conv = ConversationService.create_conversation(
            db=db,
            user_id=user_id,
            subject=subject if subject != "Tất cả môn học" else None,
            title="New Chat",
        )
        conv_id = conv.id
    st.session_state.current_conversation_id = conv_id
    if st.session_state.processor:
        st.session_state.processor.clear_history()
        logger.info(f"🧹 Cleared history for new conversation: {conv_id}")
    logger.info(f"Created new conversation: {conv_id}")
    return conv_id


def load_conversation_messages(conversation_id: int) -> list:
    with get_db() as db:
        msgs = MessageService.get_conversation_messages(db, conversation_id)
        return [
            {
                "id":         m.id,
                "role":       m.role,
                "content":    m.content,
                "created_at": m.created_at,
            }
            for m in msgs
        ]


def save_message(conversation_id: int, role: str, content: str, sources=None):
    with get_db() as db:
        MessageService.add_message(
            db=db,
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def render_sidebar(user_id: int) -> tuple[str, str, dict]:
    """
    Renders the full sidebar and returns (selected_subject, selected_model, available_models).
    All DB calls and session_state writes are UNCHANGED.
    """
    with st.sidebar:
        # Brand header with chatbot logo
        render_sidebar_header()

        # ── Subject selector ──
        st.markdown("**Môn học**")
        available_subjects = RAGProcessor.get_available_subjects()
        selected_subject = st.selectbox(
            "Môn học",
            options=["Tất cả môn học"] + available_subjects,
            key="subject_selector",
            label_visibility="collapsed",
        )

        # ── Model selector ──
        st.markdown("**Model AI**")
        available_models = RAGProcessor.get_available_llm_models()
        model_options = {
            f"{'🖥️' if info['type'] == 'local' else '☁️'} {info['name']}": key
            for key, info in available_models.items()
        }
        selected_model_display = st.selectbox(
            "Model",
            options=list(model_options.keys()),
            key="model_selector",
            label_visibility="collapsed",
        )
        selected_model = model_options[selected_model_display]

        with st.expander("ℹ️ Thông tin model", expanded=False):
            info = available_models[selected_model]
            st.caption(f"**{info['name']}** · {info['type']} · {info['provider']}")

        st.divider()

        # ── New Chat ──
        if st.button("＋  New Chat", use_container_width=True, type="primary"):
            create_new_conversation(user_id, selected_subject)
            st.rerun()

        # ── Conversation list ──
        with get_db() as db:
            conversations = ConversationService.get_user_conversations(db, user_id, limit=50)
            conv_data = [
                {
                    "id":        c.id,
                    "title":     c.title,
                    "is_pinned": c.is_pinned,
                    "subject":   c.subject,
                }
                for c in conversations
            ]

        if not conv_data:
            st.markdown(
                '<p style="font-size:12px;color:rgba(255,255,255,0.35);'
                'text-align:center;padding:16px 0 8px;">Chưa có cuộc trò chuyện</p>',
                unsafe_allow_html=True,
            )
        else:
            for conv in conv_data:
                is_active = (st.session_state.current_conversation_id == conv["id"])
                c1, c2 = st.columns([5, 1])
                with c1:
                    pin    = "📌 " if conv["is_pinned"] else ""
                    label  = f"{pin}{conv['title']}"
                    kind   = "secondary" if is_active else "tertiary"
                    if st.button(label, key=f"conv_{conv['id']}",
                                 use_container_width=True, type=kind):
                        st.session_state.current_conversation_id = conv["id"]
                        st.rerun()
                with c2:
                    if st.button("✕", key=f"del_{conv['id']}", help="Xóa"):
                        with get_db() as db:
                            ConversationService.delete_conversation(db, conv["id"], user_id)
                        if st.session_state.current_conversation_id == conv["id"]:
                            remaining = [c for c in conv_data if c["id"] != conv["id"]]
                            st.session_state.current_conversation_id = (
                                remaining[0]["id"] if remaining else None
                            )
                            if not remaining and st.session_state.processor:
                                st.session_state.processor.clear_history()
                        st.rerun()

        # ── User menu pinned at bottom ──
        render_user_menu()

    return selected_subject, selected_model, available_models


# ═════════════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ═════════════════════════════════════════════════════════════════════════════
def render_chat(user_id: int, selected_subject: str):
    """Main chat area. All data logic is UNCHANGED."""
    conv_id      = st.session_state.current_conversation_id
    conv_subject = selected_subject if selected_subject != "Tất cả môn học" else None
    messages     = []

    # Reuse latest conversation if none selected — UNCHANGED
    if not conv_id:
        with get_db() as db:
            recent = ConversationService.get_user_conversations(db, user_id, limit=1)
            if recent:
                conv_id = recent[0].id
                st.session_state.current_conversation_id = conv_id

    # Load active conversation — UNCHANGED
    if conv_id:
        with get_db() as db:
            conv = ConversationService.get_conversation(db, conv_id, user_id)
            if not conv:
                st.session_state.current_conversation_id = None
                conv_id = None
            else:
                conv_id      = conv.id
                conv_subject = conv.subject
                messages     = load_conversation_messages(conv_id)

    # ── Top header bar ──
    render_app_header()

    # ── Chat area ──
    st.markdown('<div style="padding:16px 8px 4px;">', unsafe_allow_html=True)
    render_subject_badge(conv_subject)

    if conv_id:
        # Sync history with processor — UNCHANGED
        if st.session_state.processor:
            history = [{"role": m["role"], "content": m["content"]} for m in messages]
            st.session_state.processor.set_history(history)
            logger.info(f"🔄 Synced {len(history)} messages for conversation {conv_id}")

        if messages:
            render_messages(messages)
        else:
            st.markdown(
                '<p style="color:#94a3b8;font-size:14px;'
                'padding:32px 0;text-align:center;">'
                '💡 Hãy đặt câu hỏi để bắt đầu trò chuyện</p>',
                unsafe_allow_html=True,
            )
    else:
        if st.session_state.processor:
            st.session_state.processor.clear_history()
        render_empty_state()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat input ──
    user_input = st.chat_input("Bạn muốn hỏi gì về tài liệu?")

    if user_input:
        # Lazy-create conversation on first message — UNCHANGED
        if not conv_id:
            conv_id  = create_new_conversation(user_id, selected_subject)
            messages = []

        # Save and display user message
        save_message(conv_id, "user", user_input)
        render_message("user", user_input)

        # Auto-generate title from first message — UNCHANGED
        if len(messages) == 0:
            with get_db() as db:
                ConversationService.auto_generate_title(db, conv_id, user_input)

        # Stream assistant response
        try:
            full_response = render_streaming_response(
                st.session_state.processor, user_input
            )
            save_message(conv_id, "assistant", full_response)
        except Exception as e:
            st.error(f"❌ Lỗi xử lý: {e}")
            logger.error(f"Error processing query: {e}", exc_info=True)

        inject_auto_scroll()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    init_session_state()

    if not config.validate():
        st.error("❌ Cấu hình không hợp lệ. Vui lòng kiểm tra file .env")
        st.stop()

    user_id = st.session_state.user_id

    if not os.path.exists(config.app.data_folder):
        st.error(f"📁 Không tìm thấy thư mục '{config.app.data_folder}'")
        st.stop()

    if not RAGProcessor.get_available_subjects():
        st.error(f"📚 Không tìm thấy môn học nào trong '{config.app.data_folder}'")
        st.stop()

    # Render sidebar and get current selections
    selected_subject, selected_model, available_models = render_sidebar(user_id)

    # Re-initialize processor when subject or model changes — UNCHANGED
    needs_new_processor = (
        st.session_state.processor is None
        or st.session_state.current_subject != selected_subject
        or st.session_state.current_model   != selected_model
    )
    if needs_new_processor:
        with st.spinner(f"Đang khởi tạo {selected_model}..."):
            try:
                subject_filter = (
                    None if selected_subject == "Tất cả môn học" else selected_subject
                )
                st.session_state.processor       = RAGProcessor(
                    subject=subject_filter, llm_model=selected_model
                )
                st.session_state.current_subject = selected_subject
                st.session_state.current_model   = selected_model
                logger.info(f"Processor ready — {selected_subject} / {selected_model}")
            except Exception as e:
                st.error(f"❌ Lỗi khởi tạo processor: {e}")
                logger.error(f"Processor init failed: {e}", exc_info=True)
                st.stop()

    render_chat(user_id, selected_subject)


if __name__ == "__main__":
    main()