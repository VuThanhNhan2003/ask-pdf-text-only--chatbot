import streamlit as st
from processor import RAGProcessor
import os

from dotenv import load_dotenv
load_dotenv()

# Tiêu đề
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🧠 Chatbot RAG sử dụng Gemini")

# Tải file PDF
uploaded_file = st.file_uploader("📄 Tải file PDF của bạn", type=["pdf"])

# Session state để lưu processor & lịch sử chat
if "processor" not in st.session_state:
    st.session_state.processor = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Nếu có file được tải lên
if uploaded_file is not None:
    # Lưu file PDF vào thư mục tạm thời
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Khởi tạo RAGProcessor với file PDF
    st.session_state.processor = RAGProcessor("temp_uploaded.pdf")
    st.success("✅ Đã xử lý tài liệu PDF!")

# Nếu đã xử lý xong file, hiển thị chatbox
if st.session_state.processor:
    user_input = st.chat_input("Bạn muốn hỏi gì?")

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
else:
    st.info("Vui lòng tải lên file PDF để bắt đầu.")
