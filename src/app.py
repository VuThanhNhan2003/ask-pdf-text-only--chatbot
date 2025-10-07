"""
Optimized Streamlit UI with streaming response and better UX
"""
import streamlit as st
from processor import RAGProcessor, logger
import os
from typing import List, Tuple
from config import config

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title=config.app.page_title,
    page_icon=config.app.page_icon,
    layout="wide"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-info {
        font-size: 0.9em;
        color: #666;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Chatbot RAG sử dụng Gemini - Hệ thống môn học")


# =====================================================================
# SESSION STATE MANAGEMENT
# =====================================================================
def init_session_state():
    """Initialize session state variables"""
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = None
    if "qdrant_client_cache" not in st.session_state:
        st.session_state.qdrant_client_cache = None


def get_chat_history(subject: str) -> List[Tuple[str, str]]:
    """Get chat history for a specific subject"""
    if subject not in st.session_state.chat_history:
        st.session_state.chat_history[subject] = []
    return st.session_state.chat_history[subject]


def add_to_chat_history(subject: str, speaker: str, message: str):
    """Add message to chat history"""
    if subject not in st.session_state.chat_history:
        st.session_state.chat_history[subject] = []
    st.session_state.chat_history[subject].append((speaker, message))


def clear_chat_history(subject: str = None):
    """Clear chat history"""
    if subject is None:
        st.session_state.chat_history = {}
        logger.info("🗑️ Cleared all chat history")
    else:
        st.session_state.chat_history[subject] = []
        logger.info(f"🗑️ Cleared chat history for {subject}")


# =====================================================================
# DATA VALIDATION
# =====================================================================
def validate_data_folder():
    """Validate data folder and get available subjects"""
    if not os.path.exists(config.app.data_folder):
        st.error(f"📁 Không tìm thấy thư mục '{config.app.data_folder}'. Vui lòng tạo thư mục và thêm các môn học.")
        st.stop()

    available_subjects = RAGProcessor.get_available_subjects()
    if not available_subjects:
        st.error(f"📚 Không tìm thấy môn học nào trong thư mục '{config.app.data_folder}'.")
        st.info("💡 Tạo các thư mục con trong 'data/' cho mỗi môn học và thêm file PDF vào.")
        st.stop()
    
    return available_subjects


# =====================================================================
# SIDEBAR
# =====================================================================
def render_sidebar(available_subjects: List[str]) -> str:
    """Render sidebar with controls"""
    with st.sidebar:
        st.header("📚 Chọn môn học")

        selected_subject = st.selectbox(
            "Môn học:",
            options=["Tất cả môn học"] + available_subjects,
            index=0,
            key="subject_selector",
            help="Chọn môn học để tập trung tìm kiếm"
        )

        # Subject information
        st.divider()
        render_subject_info(selected_subject, available_subjects)

        # Chat history controls
        st.divider()
        render_chat_controls(selected_subject)

        # Data management
        st.divider()
        render_data_management(selected_subject)

        # Statistics
        st.divider()
        render_statistics(selected_subject)

    return selected_subject


def render_subject_info(selected_subject: str, available_subjects: List[str]):
    """Render subject information section"""
    st.subheader("📖 Thông tin môn học")
    
    if selected_subject != "Tất cả môn học":
        subject_files = RAGProcessor.get_subject_files(selected_subject)
        st.metric("Số file PDF", len(subject_files))
        
        with st.expander("📄 Danh sách file", expanded=False):
            for i, f in enumerate(subject_files, 1):
                st.text(f"{i}. {f}")
    else:
        total_files = sum(
            len(RAGProcessor.get_subject_files(subj)) 
            for subj in available_subjects
        )
        st.metric("Tổng số file", total_files)
        st.metric("Số môn học", len(available_subjects))


def render_chat_controls(selected_subject: str):
    """Render chat history controls"""
    st.subheader("💬 Lịch sử chat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Xóa chat hiện tại", use_container_width=True):
            clear_chat_history(selected_subject)
            st.success("✅ Đã xóa chat!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Xóa tất cả", use_container_width=True):
            clear_chat_history()
            st.success("✅ Đã xóa tất cả!")
            st.rerun()


def render_data_management(selected_subject: str):
    """Render data management controls"""
    st.subheader("🗄️ Quản lý dữ liệu")
    
    if st.button("📥 Tải bổ sung dữ liệu", use_container_width=True, 
                 help="Thêm các file PDF mới vào mà không xóa dữ liệu cũ"):
        if st.session_state.processor:
            with st.spinner("Đang tải dữ liệu bổ sung..."):
                try:
                    st.session_state.processor.reload_data(clean=False)
                    st.success("✅ Đã cập nhật dữ liệu mới!")
                    logger.info(f"Data supplemented for {selected_subject}")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    logger.error(f"Failed to supplement data: {e}")

    if st.button("♻️ Tải lại toàn bộ", use_container_width=True,
                 help="Xóa và index lại toàn bộ dữ liệu từ đầu"):
        if st.session_state.processor:
            with st.spinner("Đang tải lại toàn bộ dữ liệu..."):
                try:
                    st.session_state.processor.reload_data(clean=True)
                    st.success("✅ Đã reset và index lại dữ liệu!")
                    logger.info(f"Full data reload for {selected_subject}")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    logger.error(f"Failed to reload data: {e}")


def render_statistics(selected_subject: str):
    """Render chat statistics"""
    current_chat = get_chat_history(selected_subject)
    if current_chat:
        st.subheader("📊 Thống kê")
        st.metric("Số tin nhắn", len(current_chat))
        
        # Count user vs AI messages
        user_msgs = sum(1 for speaker, _ in current_chat if speaker == "Bạn")
        ai_msgs = sum(1 for speaker, _ in current_chat if speaker == "AI")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bạn", user_msgs)
        with col2:
            st.metric("AI", ai_msgs)


# =====================================================================
# PROCESSOR INITIALIZATION
# =====================================================================
def initialize_processor(selected_subject: str) -> bool:
    """Initialize or update processor if needed"""
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
                st.success(f"✅ Đã khởi tạo cho môn: {selected_subject}")
                return True
                
            except Exception as e:
                st.error(f"❌ Lỗi khởi tạo processor: {e}")
                logger.error(f"Failed to initialize processor: {e}", exc_info=True)
                return False
    
    return True


# =====================================================================
# CHAT INTERFACE
# =====================================================================
def render_chat_interface(selected_subject: str):
    """Render main chat interface with streaming"""
    st.info(f"📖 Đang tư vấn cho: **{st.session_state.current_subject}**")

    # Display chat history first
    current_chat = get_chat_history(selected_subject)
    if current_chat:
        st.subheader("💬 Cuộc trò chuyện")
        for speaker, message in current_chat:
            if speaker == "Bạn":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message)
    else:
        st.info("💡 Hãy đặt câu hỏi để bắt đầu trò chuyện với chatbot.")

    # Chat input
    user_input = st.chat_input("Bạn muốn hỏi gì về tài liệu?")
    
    if user_input:
        # Add user message to history immediately
        add_to_chat_history(selected_subject, "Bạn", user_input)
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Stream AI response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for chunk in st.session_state.processor.get_response_stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response)
                
                # Add AI response to history
                add_to_chat_history(selected_subject, "AI", full_response)
                
            except Exception as e:
                error_msg = f"❌ Lỗi khi xử lý câu hỏi: {str(e)}"
                response_placeholder.error(error_msg)
                logger.error(f"Error processing query: {e}", exc_info=True)


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
    
    # Validate data folder
    available_subjects = validate_data_folder()
    
    # Render sidebar and get selected subject
    selected_subject = render_sidebar(available_subjects)
    
    # Initialize processor
    if not initialize_processor(selected_subject):
        st.stop()
    
    # Render chat interface
    render_chat_interface(selected_subject)
    
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