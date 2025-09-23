import os
import pymupdf4llm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Global variables để cache embedding model
_embedding_model = None
_embedding_model_path = None

def get_embedding_model():
    """Get cached embedding model"""
    global _embedding_model, _embedding_model_path
    
    if _embedding_model is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        local_model_path = "models/all-MiniLM-L6-v2"
        
        if not os.path.exists(local_model_path):
            print(f"Downloading model {model_name}...")
            os.makedirs("models", exist_ok=True)
            model = SentenceTransformer(model_name)
            model.save_pretrained(local_model_path)
            print(f"Model downloaded and saved to {local_model_path}")
        else:
            print(f"Loading model from local path: {local_model_path}")
        
        _embedding_model = SentenceTransformer(local_model_path, local_files_only=True)
        _embedding_model_path = local_model_path
    
    return _embedding_model

class RAGProcessor:
    def __init__(self, subject=None):
        """Initialize the RAG processor for a specific subject."""
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        # Use cached embedding model
        self.embedding_model = get_embedding_model()
        
        # Initialize Qdrant client (Docker service)
        self.qdrant_client = QdrantClient(host="qdrant", port=6333)
        
        # Set collection name based on subject
        if subject is None or subject == "Tất cả môn học":
            self.collection_name = "all_subjects"
            self.subject = "Tất cả môn học"
        else:
            # Convert subject name to valid collection name
            self.collection_name = self._sanitize_collection_name(subject)
            self.subject = subject
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.1
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Process documents if collection doesn't exist
        if not self._collection_exists():
            print(f"Collection '{self.collection_name}' not found. Processing documents...")
            if self.subject == "Tất cả môn học":
                self._process_all_documents()
            else:
                self._process_subject_documents(subject)
        else:
            print(f"Using existing collection: {self.collection_name}")
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Convert subject name to valid collection name"""
        # Remove Vietnamese accents and convert to lowercase
        name = name.lower()
        # Replace spaces and special chars with underscore
        name = re.sub(r'[^a-z0-9_]', '_', name)
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name or "unknown_subject"
    
    def _collection_exists(self) -> bool:
        """Check if collection exists in Qdrant."""
        try:
            collections = self.qdrant_client.get_collections()
            return any(collection.name == self.collection_name for collection in collections.collections)
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False
    
    def _create_collection(self):
        """Create Qdrant collection."""
        try:
            # Get embedding dimension
            sample_embedding = self.embedding_model.encode(["sample text"])
            embedding_dim = sample_embedding.shape[1]
            
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
            # Collection might already exist, continue
            pass
    
    def _process_all_documents(self):
        """Process all documents from /data folder."""
        data_folder = "data"
        if not os.path.exists(data_folder):
            raise ValueError("Data folder not found")
        
        # Create collection
        self._create_collection()
        
        all_chunks = []
        
        # Process each subject folder
        for subject_folder in os.listdir(data_folder):
            subject_path = os.path.join(data_folder, subject_folder)
            if os.path.isdir(subject_path):
                chunks = self._process_subject_folder(subject_folder, subject_path)
                all_chunks.extend(chunks)
        
        # Add all chunks to vector store
        if all_chunks:
            self._add_chunks_to_vectorstore(all_chunks)
            print(f"Processed total {len(all_chunks)} chunks from all subjects")
        else:
            raise ValueError("No PDF files found in data folder")
    
    def _process_subject_documents(self, subject: str):
        """Process documents for a specific subject."""
        subject_path = os.path.join("data", subject)
        if not os.path.exists(subject_path):
            raise ValueError(f"Subject folder '{subject}' not found")
        
        # Create collection
        self._create_collection()
        
        chunks = self._process_subject_folder(subject, subject_path)
        
        if chunks:
            self._add_chunks_to_vectorstore(chunks)
            print(f"Processed {len(chunks)} chunks for subject: {subject}")
        else:
            raise ValueError(f"No PDF files found for subject: {subject}")
    
    def _process_subject_folder(self, subject_name: str, subject_path: str) -> List[Dict]:
        """Process all PDF files in a subject folder."""
        chunks = []
        
        for filename in os.listdir(subject_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(subject_path, filename)
                try:
                    # Extract text using pymupdf4llm
                    md_text = pymupdf4llm.to_markdown(file_path)
                    
                    # Split text into chunks using RecursiveCharacterTextSplitter
                    text_chunks = self.text_splitter.split_text(md_text)
                    
                    # Create chunk data with metadata
                    for i, chunk_text in enumerate(text_chunks):
                        if chunk_text.strip():  # Only add non-empty chunks
                            # Estimate page number (rough approximation)
                            page_num = i // 2 + 1  # Assuming ~2 chunks per page
                            
                            chunk_data = {
                                'text': chunk_text,
                                'metadata': {
                                    'subject': subject_name,
                                    'file': filename,
                                    'page': page_num,
                                    'chunk_id': i
                                }
                            }
                            chunks.append(chunk_data)
                    
                    print(f"Processed {filename}: {len(text_chunks)} chunks")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        return chunks
    
    def _add_chunks_to_vectorstore(self, chunks: List[Dict]):
        """Add chunks to Qdrant vector store."""
        points = []
        
        for chunk in chunks:
            try:
                # Create embedding
                embedding = self.embedding_model.encode([chunk['text']])[0]
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk['text'],
                        'subject': chunk['metadata']['subject'],
                        'file': chunk['metadata']['file'],
                        'page': chunk['metadata']['page'],
                        'chunk_id': chunk['metadata']['chunk_id']
                    }
                )
                points.append(point)
                
            except Exception as e:
                print(f"Error creating embedding for chunk: {e}")
                continue
        
        # Add points to collection
        if points:
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Added {len(points)} points to collection {self.collection_name}")
            except Exception as e:
                print(f"Error adding points to collection: {e}")
    
    def _retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant chunks from vector store."""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=k
            )
            
            # Extract relevant information
            relevant_chunks = []
            for result in search_results:
                chunk_info = {
                    'text': result.payload['text'],
                    'subject': result.payload['subject'],
                    'file': result.payload['file'],
                    'page': result.payload['page'],
                    'score': result.score
                }
                relevant_chunks.append(chunk_info)
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def get_response(self, query: str) -> str:
        """Get a response for the given query."""
        try:
            # Retrieve relevant chunks
            relevant_chunks = self._retrieve_relevant_chunks(query)
            
            if not relevant_chunks:
                return "I don't know. No relevant information found in the documents."
            
            # Prepare context
            context_parts = []
            for chunk in relevant_chunks:
                source_info = f"[{chunk['subject']} - {chunk['file']} - Page {chunk['page']}]"
                context_parts.append(f"{source_info}\n{chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Simple prompt
            prompt_template = """
        Bạn là một trợ lý AI được thiết kế để cung cấp các câu trả lời phù hợp và sâu sắc dựa trên ngữ cảnh từ nhiều tài liệu khác nhau.
        Hãy sử dụng ngữ cảnh sau để trả lời câu hỏi của người dùng:

        Ngữ cảnh: {context}

        Câu hỏi của người dùng: {question}

        Câu trả lời của bạn cần:
        1. Rõ ràng, ngắn gọn và dựa trực tiếp vào ngữ cảnh đã cung cấp.
        2. Bao gồm các chi tiết cụ thể từ tài liệu khi phù hợp.
        3. Nếu có thể, hãy nêu rõ thông tin đó đến từ tài liệu nào.
        4. Nếu bạn không biết câu trả lời, hãy nói rõ và hướng dẫn nơi có thể tìm thấy thông tin nếu có thể.
        5. Nếu có nhiều tài liệu liên quan, hãy tổng hợp thông tin một cách mạch lạc.

        Câu trả lời của bạn:
        """

            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Generate response
            formatted_prompt = prompt.format(context=context, question=query)
            response = self.llm.invoke(formatted_prompt)
            
            return response.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def clear_database(self, subject: Optional[str] = None):
        """Clear database - specific collection or all collections."""
        try:
            if subject is None:
                # Clear current collection
                collection_name = self.collection_name
            elif subject == "all":
                # Clear all collections
                collections = self.qdrant_client.get_collections()
                for collection in collections.collections:
                    try:
                        self.qdrant_client.delete_collection(collection.name)
                        print(f"Deleted collection: {collection.name}")
                    except Exception as e:
                        print(f"Error deleting collection {collection.name}: {e}")
                return
            else:
                # Clear specific subject collection
                collection_name = self._sanitize_collection_name(subject)
            
            # Delete the collection
            self.qdrant_client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
            
        except Exception as e:
            print(f"Error clearing database: {e}")
    
    @staticmethod
    def get_available_subjects() -> List[str]:
        """Get list of available subjects."""
        data_folder = "data"
        if not os.path.exists(data_folder):
            return []
        
        subjects = [folder for folder in os.listdir(data_folder) 
                   if os.path.isdir(os.path.join(data_folder, folder))]
        return sorted(subjects)
    
    @staticmethod
    def get_subject_files(subject: str) -> List[str]:
        """Get list of PDF files for a specific subject."""
        subject_path = os.path.join("data", subject)
        if not os.path.exists(subject_path):
            return []
        
        pdf_files = [f for f in os.listdir(subject_path) if f.endswith('.pdf')]
        return sorted(pdf_files)
    
    def __del__(self):
        """Cleanup Qdrant client."""
        try:
            if hasattr(self, 'qdrant_client'):
                self.qdrant_client.close()
        except:
            pass