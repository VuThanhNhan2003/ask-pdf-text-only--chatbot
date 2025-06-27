import streamlit as st
from processor import RAGProcessor
import os

from dotenv import load_dotenv
load_dotenv()

# Tiêu đề
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🧠 Chatbot RAG sử dụng Gemini - Multiple PDFs")

# Tải nhiều file PDF
uploaded_files = st.file_uploader(
    "📄 Tải các file PDF của bạn", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Session state để lưu processor & lịch sử chat
if "processor" not in st.session_state:
    st.session_state.processor = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Nếu có file được tải lên
if uploaded_files:
    # Kiểm tra nếu có file mới được tải lên
    current_file_names = [f.name for f in uploaded_files]
    
    if current_file_names != st.session_state.processed_files:
        with st.spinner("Đang xử lý các tài liệu PDF... 📚"):
            # Lưu các file PDF vào thư mục tạm thời
            temp_file_paths = []
            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = f"temp_uploaded_{i}.pdf"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_file_paths.append(temp_path)
            
            # Khởi tạo RAGProcessor với nhiều file PDF
            try:
                st.session_state.processor = RAGProcessor(temp_file_paths)
                st.session_state.processed_files = current_file_names
                st.success(f"✅ Đã xử lý {len(uploaded_files)} tài liệu PDF!")
                
                # Hiển thị danh sách file đã tải
                st.info(f"📋 Các file đã tải: {', '.join(current_file_names)}")
                
            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {e}")

# Nếu đã xử lý xong file, hiển thị chatbox
if st.session_state.processor:
    user_input = st.chat_input("Bạn muốn hỏi gì về các tài liệu?")

    if user_input:
        with st.spinner("Đang suy nghĩ... 🤔"):
            try:
                response = st.session_state.processor.get_response(user_input)
                st.session_state.chat_history.append(("Bạn", user_input))
                st.session_state.chat_history.append(("AI", response))
            except Exception as e:
                st.error(f"Lỗi: {e}")

    # Hiển thị lịch sử trò chuyện
    for speaker, message in st.session_state.chat_history:
        if speaker == "Bạn":
            st.markdown(f"**🧍 {speaker}:** {message}")
        else:
            st.markdown(f"**🤖 {speaker}:** {message}")

    # Sidebar với thông tin file
    with st.sidebar:
        st.header("📁 Thông tin tài liệu")
        if st.session_state.processed_files:
            for i, filename in enumerate(st.session_state.processed_files, 1):
                st.write(f"{i}. {filename}")
        
        if st.button("🗑️ Xóa tất cả file"):
            st.session_state.processor = None
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.rerun()
else:
    st.info("Vui lòng tải lên một hoặc nhiều file PDF để bắt đầu.")