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

class RAGProcessor: # Handles the RAG pipeline processing
    def __init__(self, file_path):
        """Initialize the RAG processor with a file path."""
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        # Store the file path
        self.file_path = file_path
        
        # Initialize the conversation chain
        self.conversation_chain = self._process_document()
        
    def _process_document(self):
        """Process the document and create the conversation chain."""
        # Read the PDF file
        with open(self.file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # Create a more robust text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increase chunk size to handle larger portions of text
            chunk_overlap=300,  # Increase overlap for better context retention
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the document into chunks
        chunks = text_splitter.split_text(text)
        print(f"Document split into {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # Define a more general prompt template
        prompt_template = """
        You are an AI assistant designed to provide relevant and insightful answers based on the context provided in the document.
        Use the following context to answer the user's question:

        Context: {context}

        User's Question: {question}

        Your answer should:
        1. Be clear, concise, and directly based on the context provided.
        2. Include specific details from the document when relevant.
        3. State if you don't know the answer, and provide guidance on where to find the information if possible.

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
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        return conversation_chain
    
    def get_response(self, query):
        """Get a response for the given query."""
        response = self.conversation_chain.invoke({"question": query})
        return response['answer']
