import os
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load API keys from .env file
from dotenv import load_dotenv
load_dotenv()

class RAGProcessor: # Handles the RAG pipeline processing for multiple PDFs
    def __init__(self, file_paths):
        """Initialize the RAG processor with multiple file paths."""
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        # Store the file paths (can be a single file or list of files)
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
        
        # Initialize the conversation chain
        self.conversation_chain = self._process_documents()
        
    def _process_documents(self):
        """Process multiple documents and create the conversation chain."""
        all_text = ""
        document_info = []
        
        # Read all PDF files
        for i, file_path in enumerate(self.file_paths):
            try:
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    doc_text = ""
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        doc_text += page_text
                    
                    # Add document identifier to the text
                    doc_name = os.path.basename(file_path)
                    doc_text_with_source = f"\n[DOCUMENT: {doc_name}]\n{doc_text}\n[END OF DOCUMENT: {doc_name}]\n"
                    all_text += doc_text_with_source
                    
                    document_info.append({
                        'name': doc_name,
                        'path': file_path,
                        'pages': len(reader.pages)
                    })
                    
                    print(f"Processed {doc_name}: {len(reader.pages)} pages")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_text.strip():
            raise ValueError("No text could be extracted from the provided PDF files")
        
        # Create a more robust text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increase chunk size to handle larger portions of text
            chunk_overlap=300,  # Increase overlap for better context retention
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the combined document into chunks
        chunks = text_splitter.split_text(all_text)
        print(f"All documents split into {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # Define a more general prompt template for multiple documents
        prompt_template = """
        You are an AI assistant designed to provide relevant and insightful answers based on the context from multiple documents.
        Use the following context to answer the user's question:

        Context: {context}

        User's Question: {question}

        Your answer should:
        1. Be clear, concise, and directly based on the context provided.
        2. Include specific details from the documents when relevant.
        3. When possible, mention which document the information comes from.
        4. State if you don't know the answer, and provide guidance on where to find the information if possible.
        5. If information from multiple documents is relevant, synthesize it coherently.

        Your Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize the model
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        
        # Create the conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),  # Increase k for multiple docs
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        # Store document info for reference
        self.document_info = document_info
        
        return conversation_chain
    
    def get_response(self, query):
        """Get a response for the given query."""
        response = self.conversation_chain.invoke({"question": query})
        return response['answer']
    
    def get_document_info(self):
        """Get information about processed documents."""
        return self.document_info