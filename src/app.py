import streamlit as st
from .processor import RAGProcessor
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🧠 Chatbot RAG sử dụng Gemini - Hệ thống môn học")

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}  # Dict to store chat history by subject
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = None

def get_chat_history(subject: str) -> List[Tuple[str, str]]:
    """Get chat history for specific subject"""
    if subject not in st.session_state.chat_history:
        st.session_state.chat_history[subject] = []
    return st.session_state.chat_history[subject]

def add_to_chat_history(subject: str, speaker: str, message: str):
    """Add message to chat history for specific subject"""
    if subject not in st.session_state.chat_history:
        st.session_state.chat_history[subject] = []
    st.session_state.chat_history[subject].append((speaker, message))

def clear_chat_history(subject: str = None):
    """Clear chat history for specific subject or all"""
    if subject is None:
        st.session_state.chat_history = {}
    else:
        if subject in st.session_state.chat_history:
            st.session_state.chat_history[subject] = []

# Initialize session state
init_session_state()

# Check data folder
data_folder = "data"
if not os.path.exists(data_folder):
    st.error("📁 Không tìm thấy thư mục 'data'. Vui lòng tạo thư mục và thêm các môn học.")
    st.info("Cấu trúc thư mục: data/<tên_môn_học>/<file.pdf>")
    st.stop()

# Get available subjects
available_subjects = RAGProcessor.get_available_subjects()

if not available_subjects:
    st.error("📚 Không tìm thấy môn học nào trong thư mục 'data'.")
    st.info("Cấu trúc thư mục: data/<tên_môn_học>/<file.pdf>")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📚 Chọn môn học")
    
    # Subject selection
    selected_subject = st.selectbox(
        "Môn học:",
        options=["Tất cả môn học"] + available_subjects,
        index=0,
        key="subject_selector"
    )
    
    # Display subject info
    if selected_subject != "Tất cả môn học":
        subject_files = RAGProcessor.get_subject_files(selected_subject)
        if subject_files:
            st.write(f"📄 **{len(subject_files)} files:**")
            for i, filename in enumerate(subject_files, 1):
                st.write(f"{i}. {filename}")
        else:
            st.write("📄 **Không có file PDF nào**")
    else:
        total_files = 0
        for subject in available_subjects:
            subject_files = RAGProcessor.get_subject_files(subject)
            total_files += len(subject_files)
        st.write(f"📄 **Tổng cộng: {total_files} files từ {len(available_subjects)} môn học**")
    
    st.divider()
    
    # Chat history controls
    st.subheader("💬 Lịch sử chat")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xóa chat hiện tại", use_container_width=True):
            clear_chat_history(selected_subject)
            if hasattr(st.session_state, 'processor') and st.session_state.processor:
                st.session_state.processor.memory.clear()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Xóa tất cả chat", use_container_width=True):
            clear_chat_history()
            if hasattr(st.session_state, 'processor') and st.session_state.processor:
                st.session_state.processor.memory.clear()
            st.rerun()
    
    st.divider()
    
    # Database controls
    st.subheader("🗄️ Quản lý Database")
    
    if st.button("🔄 Tải lại môn học hiện tại", use_container_width=True):
        if hasattr(st.session_state, 'processor') and st.session_state.processor:
            try:
                st.session_state.processor.clear_database()
                st.session_state.processor = None
                st.session_state.current_subject = None
                st.success("✅ Đã xóa và sẽ tải lại dữ liệu!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi xóa database: {e}")
    
    if st.button("🗄️ Xóa toàn bộ database", use_container_width=True):
        if hasattr(st.session_state, 'processor') and st.session_state.processor:
            try:
                st.session_state.processor.clear_database("all")
                st.session_state.processor = None
                st.session_state.current_subject = None
                clear_chat_history()
                st.success("✅ Đã xóa toàn bộ database!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi xóa database: {e}")
    
    # Display current chat history count
    current_chat = get_chat_history(selected_subject)
    if current_chat:
        st.info(f"💬 {len(current_chat)} tin nhắn trong cuộc trò chuyện này")

# Main content area
# Check if processor needs to be initialized or changed
needs_new_processor = (
    st.session_state.processor is None or 
    st.session_state.current_subject != selected_subject
)

if needs_new_processor:
    with st.spinner("Đang tải dữ liệu... 📚"):
        try:
            # Clear old processor
            if st.session_state.processor:
                try:
                    del st.session_state.processor
                except:
                    pass
            
            # Create new processor
            if selected_subject == "Tất cả môn học":
                st.session_state.processor = RAGProcessor()
                st.session_state.current_subject = "Tất cả môn học"
                st.success("✅ Đã tải tất cả môn học!")
            else:
                st.session_state.processor = RAGProcessor(subject=selected_subject)
                st.session_state.current_subject = selected_subject
                st.success(f"✅ Đã tải môn học: {selected_subject}")
                
        except Exception as e:
            st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
            st.stop()

# Display current subject
if st.session_state.processor:
    st.info(f"📖 Đang tư vấn cho: **{st.session_state.current_subject}**")
    
    # Chat interface
    user_input = st.chat_input("Bạn muốn hỏi gì về tài liệu?", key="chat_input")

    if user_input:
        with st.spinner("Đang suy nghĩ... 🤔"):
            try:
                response = st.session_state.processor.get_response(user_input)
                add_to_chat_history(selected_subject, "Bạn", user_input)
                add_to_chat_history(selected_subject, "AI", response)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý câu hỏi: {e}")

    # Display chat history for current subject
    current_chat = get_chat_history(selected_subject)
    if current_chat:
        st.subheader("💬 Cuộc trò chuyện")
        
        for speaker, message in current_chat:
            if speaker == "Bạn":
                with st.chat_message("user"):
                    st.markdown(message)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message)
    
    # Instructions (only show if no chat history)
    if not current_chat:
        st.markdown("""
        ### 📋 Hướng dẫn sử dụng:
        1. **Chọn môn học** từ sidebar bên trái
        2. **Đặt câu hỏi** trong ô chat bên dưới
        3. **Nhận câu trả lời** dựa trên tài liệu đã được xử lý
        
        ### 💡 Mẹo sử dụng:
        - Đặt câu hỏi cụ thể để nhận được câu trả lời chính xác
        - Hệ thống sẽ trả lời dựa trên nội dung tài liệu đã tải
        - Chọn "Tất cả môn học" để tìm kiếm trong toàn bộ tài liệu
        
        ### 📁 Cấu trúc thư mục:
        ```
        data/
        ├── Toán học/
        │   ├── file01.pdf
        │   └── file02.pdf
        ├── Vật lý/
        │   ├── file01.pdf
        │   └── file02.pdf
        └── ...
        ```
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🤖 RAG Chatbot với Streamlit + Qdrant + Gemini
    </div>
    """, 
    unsafe_allow_html=True
)