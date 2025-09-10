from processor import RAGProcessor

processor = RAGProcessor()
subjects = processor.get_available_subjects()
print("Các môn học được phát hiện:", subjects)
