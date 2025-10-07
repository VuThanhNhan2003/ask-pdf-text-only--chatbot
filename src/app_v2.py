"""
Enhanced Streamlit UI with authentication and conversation management
"""
import streamlit as st
from processor import RAGProcessor, logger
import os
from typing import List, Tuple
from config import config
from datetime import datetime

# Import new services
from components.auth_ui import check_authentication, render_user_menu, logout
from database.database import get_db, init_db
from services.conversation_service import ConversationService, MessageService

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title=config.app.page_title,
    page_icon=config.app.page_icon,
    layout="wide"
)

# Initialize database on first run
try:
    init_db()
except:
    pass

# Check authentication first
check_authentication()

# Custom CSS
st.markdown("""
<style>
    .conversation-item {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.3rem;
        cursor: pointer;
        border-left: 3px solid transparent;
    }
    .conversation-item:hover {
        background-color: #f0f2f6;
        border-left-color: #1f77b4;
    }
    .conversation-item.active {
        background-color: #e6f2ff;
        border-left-color: #1f77b4;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 RAG Chatbot - Hệ thống môn học")


# =====================================================================
# SESSION STATE MANAGEMENT
# =====================================================================
def init_session_state():
    """Initialize session state variables"""
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = None


# =====================================================================
# CONVERSATION MANAGEMENT
# =====================================================================
def create_new_conversation(user_id: int, subject: str):
    """Create new conversation"""
    with get_db() as db:
        conv = ConversationService.create_conversation(
            db=db,
            user_id=user_id,
            subject=subject if subject != "Tất cả môn học" else None,
            title="New Chat"
        )
        conv_id = conv.id  # Extract ID while session is active
        
    st.session_state.current_conversation_id = conv_id
    logger.info(f"Created new conversation: {conv_id}")
    return conv_id


def load_conversation_messages(conversation_id: int):
    """Load messages for a conversation"""
    with get_db() as db:
        messages = MessageService.get_conversation_messages(db, conversation_id)
        
        # Extract message data while session is active
        messages_data = []
        for msg in messages:
            messages_data.append({
                'id': msg.id,
                'role': msg.role,
                'content': msg.content,
                'created_at': msg.created_at
            })
    
    return messages_data


def save_message(conversation_id: int, role: str, content: str, sources=None):
    """Save message to database"""
    with get_db() as db:
        return MessageService.add_message(
            db=db,
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources
        )


# =====================================================================
# SIDEBAR - CONVERSATION LIST
# =====================================================================
def render_conversations_sidebar(user_id: int, current_subject: str):
    """Render conversation list in sidebar"""
    with st.sidebar:
        st.header("💬 Conversations")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_conversation(user_id, current_subject)
            st.rerun()
        
        st.divider()
        
        # Load conversations
        with get_db() as db:
            conversations = ConversationService.get_user_conversations(db, user_id, limit=50)
            
            # Extract all data while session is active
            conv_data = []
            for conv in conversations:
                conv_data.append({
                    'id': conv.id,
                    'title': conv.title,
                    'is_pinned': conv.is_pinned,
                    'subject': conv.subject
                })
        
        if not conv_data:
            st.info("Chưa có cuộc trò chuyện nào")
        else:
            # Display conversations using extracted data
            for conv in conv_data:
                is_active = (st.session_state.current_conversation_id == conv['id'])
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Conversation button
                    label = f"{'📌 ' if conv['is_pinned'] else ''}{conv['title']}"
                    if st.button(
                        label,
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type="secondary" if is_active else "tertiary"
                    ):
                        st.session_state.current_conversation_id = conv['id']
                        st.rerun()
                
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"del_{conv['id']}"):
                        with get_db() as db:
                            ConversationService.delete_conversation(db, conv['id'], user_id)
                        if st.session_state.current_conversation_id == conv['id']:
                            st.session_state.current_conversation_id = None
                        st.rerun()


# =====================================================================
# MAIN CHAT INTERFACE
# =====================================================================
def render_chat_interface(user_id: int, selected_subject: str):
    """Render main chat interface"""
    
    # Get or create conversation
    if not st.session_state.current_conversation_id:
        conv_id = create_new_conversation(user_id, selected_subject)
        conv_subject = selected_subject if selected_subject != "Tất cả môn học" else None
        conv_title = "New Chat"
    else:
        with get_db() as db:
            conv = ConversationService.get_conversation(
                db, st.session_state.current_conversation_id, user_id
            )
            if not conv:
                # Conversation deleted or doesn't exist
                st.session_state.current_conversation_id = None
                st.rerun()
            
            # Extract data while session is active
            conv_id = conv.id
            conv_subject = conv.subject
            conv_title = conv.title
    
    # Display conversation info (using extracted data, not ORM object)
    st.info(f"📖 Môn học: **{conv_subject or 'Tất cả môn học'}**")
    
    # Load and display messages
    messages = load_conversation_messages(conv_id)
    
    if messages:
        for msg in messages:
            # Use extracted data (dict) not ORM object
            avatar = "user" if msg['role'] == "user" else "assistant"
            
            with st.chat_message(avatar):
                st.markdown(msg['content'])
    else:
        st.info("💡 Hãy đặt câu hỏi để bắt đầu trò chuyện")
    
    # Chat input
    user_input = st.chat_input("Bạn muốn hỏi gì về tài liệu?")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Save user message
        save_message(conv_id, "user", user_input)
        
        # Auto-generate title for first message
        if len(messages) == 0:
            with get_db() as db:
                ConversationService.auto_generate_title(db, conv_id, user_input)
        
        # Generate AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for chunk in st.session_state.processor.get_response_stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response)
                
                # Save AI response
                save_message(conv_id, "assistant", full_response)
                
            except Exception as e:
                error_msg = f"❌ Lỗi: {str(e)}"
                response_placeholder.error(error_msg)
                logger.error(f"Error processing query: {e}", exc_info=True)
        
        st.rerun()


# =====================================================================
# MAIN APP
# =====================================================================
def main():
    """Main application logic"""
    # Initialize session state
    init_session_state()
    
    # Validate configuration
    if not config.validate():
        st.error("❌ Cấu hình không hợp lệ. Vui lòng kiểm tra file .env")
        st.stop()
    
    # Get user ID
    user_id = st.session_state.user_id
    
    # Validate data folder
    if not os.path.exists(config.app.data_folder):
        st.error(f"📁 Không tìm thấy thư mục '{config.app.data_folder}'")
        st.stop()
    
    available_subjects = RAGProcessor.get_available_subjects()
    if not available_subjects:
        st.error(f"📚 Không tìm thấy môn học nào trong '{config.app.data_folder}'")
        st.stop()
    
    # Subject selector in sidebar
    with st.sidebar:
        st.header("📚 Chọn môn học")
        selected_subject = st.selectbox(
            "Môn học:",
            options=["Tất cả môn học"] + available_subjects,
            key="subject_selector"
        )
        
        st.divider()
    
    # Initialize processor if needed
    needs_new_processor = (
        st.session_state.processor is None or
        st.session_state.current_subject != selected_subject
    )
    
    if needs_new_processor:
        with st.spinner("Đang khởi tạo processor..."):
            try:
                subject_filter = None if selected_subject == "Tất cả môn học" else selected_subject
                st.session_state.processor = RAGProcessor(subject_filter)
                st.session_state.current_subject = selected_subject
                logger.info(f"Processor initialized for: {selected_subject}")
            except Exception as e:
                st.error(f"❌ Lỗi khởi tạo: {e}")
                st.stop()
    
    # Render conversations sidebar
    render_conversations_sidebar(user_id, selected_subject)
    
    # Render user menu at bottom of sidebar
    render_user_menu()
    
    # Render main chat interface
    render_chat_interface(user_id, selected_subject)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 RAG Chatbot với Streamlit + Qdrant + Gemini | "
        f"📝 Logs: {config.app.log_folder}/"
        "</div>",
        unsafe_allow_html=True
    )


# =====================================================================
# RUN APP
# =====================================================================
if __name__ == "__main__":
    main()