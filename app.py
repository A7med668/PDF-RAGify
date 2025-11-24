%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import torch
import time
import warnings
import tempfile
import os
import re
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Chat with PDF - Enhanced Version",
    page_icon="📚",
    layout="wide"
)

class PDFQASystem:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.initialized = False
        self.pdf_name = None
        self.full_text = ""
        self.llm_client = None
        self.api_key = None
    
    def set_api_key(self, api_key):
        """Set API key"""
        self.api_key = api_key
        if api_key:
            try:
                from google import genai
                self.llm_client = genai.Client(api_key=api_key)
                return True
            except Exception as e:
                st.error(f"❌ Error initializing Gemini: {e}")
                return False
        return False
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PdfReader(pdf_file)
            full_text = ""
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            self.full_text = full_text
            return full_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    
    def chunk_text(self, text, chunk_size=800, overlap=100):
        """Split text into chunks using LangChain"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            st.warning(f"Using basic chunking method: {e}")
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + '. '
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + '. '
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
    
    def embed_chunks(self, chunks):
        """Create embeddings for text chunks using multilingual model"""
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        embeddings = model.encode(chunks, convert_to_numpy=True)
        return model, embeddings
    
    def create_faiss_index(self, embeddings):
        """Create FAISS index for semantic search"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    
    def search_index(self, query, k=5):
        """Search for relevant chunks using semantic similarity"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                relevant_chunks.append(self.chunks[idx])
        
        return relevant_chunks
    
    def initialize_system(self, pdf_file, pdf_name):
        """Initialize the Q&A system with uploaded PDF"""
        try:
            text = self.extract_text_from_pdf(pdf_file)
            
            if text is None or len(text.strip()) == 0:
                st.error("Could not extract text from PDF. The file might be scanned or corrupted.")
                return False
            
            with st.spinner("🔄 Splitting text into chunks..."):
                self.chunks = self.chunk_text(text)
            
            with st.spinner("🧠 Creating embeddings..."):
                self.embedding_model, embeddings = self.embed_chunks(self.chunks)
            
            with st.spinner("🔍 Building search index..."):
                self.index = self.create_faiss_index(embeddings)
            
            self.initialized = True
            self.pdf_name = pdf_name
            return True
            
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            return False
    
    def _generate_llm_answer(self, question, context):
        """Generate coherent answer using large language model"""
        if self.llm_client is None:
            clean_answer = self._create_fallback_answer(context).replace('\\n', '\n').replace('\n\n', '\n')
            return f"💡 **Answer from text search:**\n\n{clean_answer}"
        
        try:
            prompt = f"""
            You are an intelligent assistant specialized in analyzing documents and answering questions based on the information in them.
            
            Task:
            - Read the provided context carefully
            - Answer the question based only on the information in the context
            - If information is insufficient, state that clearly
            - Use clear and understandable language
            - Structure the answer logically
            
            Question: {question}
            
            Context from document:
            {context}
            
            Answer:
            """
            
            response = self.llm_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            clean_response = response.text.replace('\\n', '\n').replace('\n\n', '\n')
            return f"🤖 **Smart answer from language model:**\n\n{clean_response}"
            
        except Exception as e:
            clean_answer = self._create_fallback_answer(context).replace('\\n', '\n').replace('\n\n', '\n')
            return f"💡 **Answer from text search:**\n\n{clean_answer}"
    
    def _create_fallback_answer(self, context):
        """Create alternative answer when LLM is not available"""
        clean_context = context.replace('\\n', ' ').replace('\n', ' ')
        sentences = clean_context.split('. ')
        if len(sentences) > 4:
            return ". ".join(sentences[:4]) + "..."
        else:
            return clean_context
    
    def answer_question(self, question):
        """Generate answer for a given question"""
        if not self.initialized:
            return "System not initialized. Please upload a PDF file first.", []
        
        try:
            top_chunks = self.search_index(question, k=5)
            context = " ".join(top_chunks)
            
            if not context.strip():
                return "❌ Could not find relevant information in the document.", []
            
            answer = self._generate_llm_answer(question, context)
            
            return answer, top_chunks
            
        except Exception as e:
            return f"Error generating answer: {str(e)}", []

@st.cache_resource
def load_qa_system():
    return PDFQASystem()

def main():
    st.title("📚 Chat with PDF - Enhanced Version")
    st.markdown("### Upload a PDF and ask questions about its content")
    
    qa_system = load_qa_system()
    
    with st.sidebar:
        st.markdown("### 🔑 AI Settings (Optional)")
        
        api_key = st.text_input(
            "Google AI API Key (Optional)",
            type="password",
            placeholder="Enter API key to activate AI...",
            help="Get free key from: https://aistudio.google.com/app/apikey"
        )
        
        if api_key:
            if st.button("Activate AI"):
                if qa_system.set_api_key(api_key):
                    st.success("✅ AI activated successfully!")
                else:
                    st.error("❌ Failed to activate AI. Check your key.")
        
        st.markdown("---")
        st.markdown("### 📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                success = qa_system.initialize_system(uploaded_file, uploaded_file.name)
                
                if success:
                    st.success(f"✅ PDF processed successfully!")
                    st.info(f"**File:** {uploaded_file.name}")
                    st.info(f"**Chunks processed:** {len(qa_system.chunks)}")
                    
                    text_sample = qa_system.chunks[0][:300] + "..." if qa_system.chunks else "No text extracted"
                    with st.expander("📊 Document Preview"):
                        st.text_area("First chunk sample:", text_sample, height=120)
                else:
                    st.error("Failed to process PDF. Please try another file.")
        
        st.markdown("---")
        st.markdown("### 💡 Tips for Better Answers")
        st.markdown("""
        **Without API Key:**
        - System will rely on precise text search
        - Answers will be directly from the document
        
        **With API Key:**
        - Smarter and more coherent answers
        - Better context understanding
        - Information summarization
        """)
        
        st.markdown("---")
        st.markdown("### 🛠️ System Information")
        st.info(f"""
        **Models Used:**
        - Embedding: multilingual-mpnet-base-v2
        - AI: {'Activated' if qa_system.llm_client else 'Not activated'}
        - Search: FAISS
        - Chunking: LangChain
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Chat")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("🔍 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source[:400] + "..." if len(source) > 400 else source)
                            st.markdown("---")
        
        if prompt := st.chat_input("Ask a question about your PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            if not qa_system.initialized:
                response = "Please upload a PDF file first to start chatting."
                sources = []
            else:
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching document..."):
                        start_time = time.time()
                        response, sources = qa_system.answer_question(prompt)
                        response_time = time.time() - start_time
                    
                    st.markdown(response)
                    st.caption(f"⏱️ Response time: {response_time:.2f}s")
                    
                    if sources and "Could not find" not in response and "❌" not in response:
                        with st.expander("🔍 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(source[:400] + "..." if len(source) > 400 else source)
                                st.markdown("---")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📊 Document Info", use_container_width=True) and qa_system.initialized:
            st.info(f"""
            **Document Analysis:**
            - PDF: {qa_system.pdf_name}
            - Text Chunks: {len(qa_system.chunks)}
            - Search Engine: FAISS
            - Embedding Model: Multilingual
            - AI: {'Activated 🟢' if qa_system.llm_client else 'Not activated 🔴'}
            """)
        
        st.markdown("---")
        st.markdown("### 🚀 Sample Questions")
        st.markdown("""
        **Financial/Budget:**
        - "What is the research budget?"
        - "How much funding for research?"
        
        **Strategic:**
        - "What are the main priorities?"
        - "Strategic focus areas?"
        
        **General:**
        - "What is the main objective?"
        - "Explain the key points"
        """)

if __name__ == "__main__":
    main()