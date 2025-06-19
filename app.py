# =============================================================================
# STREAMLIT PDF RAG APPLICATION
# Web-based PDF Q&A system with visual page retrieval
# =============================================================================

import streamlit as st
import os
import tempfile
import PyPDF2
import io
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import Image
import sqlite3
import pandas as pd
from datetime import datetime

# Streamlit imports
from streamlit_option_menu import option_menu

# LangChain imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.docstore.document import Document
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    import openai
except ImportError as e:
    st.error(f"Missing required packages. Please install: {e}")
    st.stop()

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="PDF RAG Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# STREAMLIT PDF RAG CLASS
# =============================================================================

class StreamlitPDFRAG:
    """Streamlit-based PDF RAG system"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'vector_db' not in st.session_state:
            st.session_state.vector_db = None
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'pdf_images' not in st.session_state:
            st.session_state.pdf_images = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'chunks' not in st.session_state:
            st.session_state.chunks = []
        if 'pdf_filename' not in st.session_state:
            st.session_state.pdf_filename = None
    
    def extract_pdf_text(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Extract text using PyPDF2
            with open(tmp_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                page_data = []
                all_text = ""
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    page_data.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                    all_text += f"\\n--- PAGE {page_num + 1} ---\\n" + page_text
            
            # Convert PDF to images
            try:
                pdf_images = convert_from_path(tmp_path)
                st.session_state.pdf_images = pdf_images
            except Exception as e:
                st.warning(f"Could not convert PDF to images: {e}")
                st.session_state.pdf_images = None
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return all_text, page_data, num_pages
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None, None, 0
    
    def create_chunks(self, page_data, pdf_filename):
        """Create text chunks with metadata"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = []
        
        for page_info in page_data:
            page_num = page_info['page_number']
            page_text = page_info['text']
            
            if len(page_text.strip()) < 50:
                continue
            
            page_chunks = text_splitter.split_text(page_text)
            
            for i, chunk in enumerate(page_chunks):
                chunks.append({
                    'content': chunk,
                    'page_number': page_num,
                    'chunk_id': f"page_{page_num}_chunk_{i+1}",
                    'source_file': pdf_filename,
                    'char_count': len(chunk)
                })
        
        return chunks
    
    def create_vector_database(self, chunks, api_key):
        """Create vector database with embeddings"""
        try:
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-3-small"
            )
            
            # Convert chunks to documents
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        'page_number': chunk['page_number'],
                        'chunk_id': chunk['chunk_id'],
                        'source_file': chunk['source_file'],
                        'char_count': chunk['char_count']
                    }
                )
                documents.append(doc)
            
            # Create vector database
            vector_db = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=f"./vectordb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            return vector_db
            
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return None
    
    def create_rag_system(self, vector_db, api_key):
        """Create RAG system with custom prompt"""
        try:
            # Custom prompt template
            prompt_template = """
You are a business analyst answering questions about a document.

Use the following context to answer the question. ALWAYS include page references.

Context: {context}
Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. ALWAYS mention page numbers like [Page X]
3. Structure answers professionally for business reporting
4. Include specific numbers, percentages, and metrics when available
5. If information is not in context, say so clearly

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            # Initialize ChatGPT
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model="gpt-4o-mini",
                temperature=0.1
            )
            
            # Create RAG chain
            rag_system = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            return rag_system
            
        except Exception as e:
            st.error(f"Error creating RAG system: {str(e)}")
            return None
    
    def find_relevant_pages(self, question, vector_db, top_k=3):
        """Find most relevant pages for a question"""
        try:
            relevant_docs = vector_db.similarity_search(question, k=15)
            
            page_scores = {}
            for doc in relevant_docs:
                page_num = doc.metadata['page_number']
                if page_num not in page_scores:
                    page_scores[page_num] = {'score': 0, 'chunks': []}
                page_scores[page_num]['score'] += 1
                page_scores[page_num]['chunks'].append(doc.page_content[:200] + "...")
            
            sorted_pages = sorted(page_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
            return sorted_pages
        except Exception as e:
            st.error(f"Error finding relevant pages: {str(e)}")
            return []

# =============================================================================
# STREAMLIT APP INTERFACE
# =============================================================================

def main():
    """Main Streamlit application"""
    
    # Initialize the PDF RAG system
    pdf_rag = StreamlitPDFRAG()
    
    # Header
    st.markdown('<h1 class="main-header">📄 PDF RAG Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("**Upload a PDF, ask questions, and get answers with visual page references!**")
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        selected = option_menu(
            menu_title=None,
            options=["📤 Upload PDF", "🤖 Ask Questions", "📊 Analytics", "ℹ️ About"],
            icons=["upload", "chat-dots", "bar-chart", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )
        
        # API Key input in sidebar
        st.markdown("### 🔑 OpenAI API Key")
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        if api_key:
            if api_key.startswith('sk-'):
                st.success("✅ API key format looks correct!")
            else:
                st.error("❌ API key should start with 'sk-'")
    
    # Main content area
    if selected == "📤 Upload PDF":
        upload_pdf_section(pdf_rag, api_key)
    
    elif selected == "🤖 Ask Questions":
        ask_questions_section(pdf_rag, api_key)
    
    elif selected == "📊 Analytics":
        analytics_section(pdf_rag)
    
    elif selected == "ℹ️ About":
        about_section()

def upload_pdf_section(pdf_rag, api_key):
    """PDF Upload Section"""
    st.markdown('<h2 class="sub-header">📤 Upload and Process PDF</h2>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        st.session_state.pdf_filename = uploaded_file.name
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📄 **File:** {uploaded_file.name}")
            st.info(f"📊 **Size:** {len(uploaded_file.getvalue())} bytes")
        
        with col2:
            process_button = st.button("🚀 Process PDF", type="primary")
        
        if process_button:
            if not api_key:
                st.error("❌ Please enter your OpenAI API key in the sidebar first!")
                return
            
            with st.spinner("Processing PDF..."):
                # Extract text
                st.info("📖 Extracting text from PDF...")
                pdf_text, page_data, num_pages = pdf_rag.extract_pdf_text(uploaded_file)
                
                if pdf_text:
                    st.success(f"✅ Extracted text from {num_pages} pages")
                    
                    # Create chunks
                    st.info("📝 Creating text chunks...")
                    chunks = pdf_rag.create_chunks(page_data, uploaded_file.name)
                    st.session_state.chunks = chunks
                    st.success(f"✅ Created {len(chunks)} text chunks")
                    
                    # Create vector database
                    st.info("🧠 Creating vector embeddings...")
                    vector_db = pdf_rag.create_vector_database(chunks, api_key)
                    
                    if vector_db:
                        st.session_state.vector_db = vector_db
                        st.success("✅ Vector database created successfully!")
                        
                        # Create RAG system
                        st.info("🤖 Setting up Q&A system...")
                        rag_system = pdf_rag.create_rag_system(vector_db, api_key)
                        
                        if rag_system:
                            st.session_state.rag_system = rag_system
                            st.session_state.pdf_processed = True
                            
                            st.markdown('<div class="success-box">🎉 <strong>PDF processed successfully!</strong><br>You can now ask questions in the "Ask Questions" section.</div>', unsafe_allow_html=True)
                            
                            # Show processing summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📄 Pages", num_pages)
                            with col2:
                                st.metric("📝 Chunks", len(chunks))
                            with col3:
                                st.metric("🧠 Vectors", len(chunks))
                        else:
                            st.error("❌ Failed to create RAG system")
                    else:
                        st.error("❌ Failed to create vector database")
                else:
                    st.error("❌ Failed to extract text from PDF")

def ask_questions_section(pdf_rag, api_key):
    """Questions Section"""
    st.markdown('<h2 class="sub-header">🤖 Ask Questions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.pdf_processed:
        st.warning("⚠️ Please upload and process a PDF first!")
        return
    
    # Sample questions
    with st.expander("💡 Sample Questions", expanded=True):
        sample_questions = [
            "What is the financial performance?",
            "What are the key metrics and KPIs?",
            "What challenges are mentioned?",
            "Show me market analysis information",
            "What are the main conclusions?",
            "Tell me about performance trends"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            col = cols[i % 2]
            if col.button(f"📝 {question}", key=f"sample_{i}"):
                st.session_state.current_question = question
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Ask anything about your PDF document...",
        value=st.session_state.get('current_question', '')
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary")
    
    if ask_button and question.strip():
        with st.spinner("Searching for relevant information..."):
            try:
                # Get AI answer
                result = st.session_state.rag_system.invoke({"query": question})
                ai_answer = result['result']
                source_docs = result['source_documents']
                
                # Display answer
                st.markdown("### 💬 Answer:")
                st.markdown(ai_answer)
                
                # Show source pages
                source_pages = sorted(list(set([doc.metadata['page_number'] for doc in source_docs])))
                st.info(f"📚 **Answer based on pages:** {source_pages}")
                
                # Find and display relevant visual pages
                if st.session_state.pdf_images:
                    st.markdown("### 🖼️ Relevant Visual Pages:")
                    
                    relevant_pages = pdf_rag.find_relevant_pages(question, st.session_state.vector_db, top_k=3)
                    
                    if relevant_pages:
                        for page_num, page_info in relevant_pages:
                            if page_num <= len(st.session_state.pdf_images):
                                st.markdown(f"#### 📄 Page {page_num} (Relevance: {page_info['score']} chunks)")
                                
                                # Display PDF page image
                                img = st.session_state.pdf_images[page_num - 1]
                                st.image(img, caption=f"Page {page_num}", use_column_width=True)
                                
                                # Show relevant text snippets
                                with st.expander(f"📝 Relevant content from Page {page_num}"):
                                    for i, chunk in enumerate(page_info['chunks'][:2], 1):
                                        st.write(f"**Snippet {i}:** {chunk}")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'question': question,
                    'answer': ai_answer,
                    'pages': source_pages
                })
                
                st.success(f"✅ Question processed! ({len(st.session_state.chat_history)} questions in history)")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")

def analytics_section(pdf_rag):
    """Analytics Section"""
    st.markdown('<h2 class="sub-header">📊 Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.pdf_processed:
        st.warning("⚠️ Please upload and process a PDF first!")
        return
    
    # Document statistics
    st.markdown("### 📄 Document Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 PDF Pages", len(st.session_state.pdf_images) if st.session_state.pdf_images else 0)
    with col2:
        st.metric("📝 Text Chunks", len(st.session_state.chunks))
    with col3:
        st.metric("🧠 Vector Embeddings", len(st.session_state.chunks))
    with col4:
        st.metric("❓ Questions Asked", len(st.session_state.chat_history))
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        
        # Convert to DataFrame for better display
        history_df = pd.DataFrame(st.session_state.chat_history)
        
        for i, row in history_df.iterrows():
            with st.expander(f"❓ {row['question'][:50]}... ({row['timestamp']})"):
                st.write(f"**Question:** {row['question']}")
                st.write(f"**Answer:** {row['answer'][:200]}...")
                st.write(f"**Pages Referenced:** {row['pages']}")
    
    # Chunk analysis
    if st.session_state.chunks:
        st.markdown("### 📝 Chunk Analysis")
        
        # Create DataFrame from chunks
        chunks_df = pd.DataFrame(st.session_state.chunks)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Page distribution
            page_counts = chunks_df['page_number'].value_counts().sort_index()
            st.bar_chart(page_counts)
            st.caption("Chunks per page")
        
        with col2:
            # Character count distribution
            char_counts = chunks_df['char_count']
            st.line_chart(char_counts.reset_index())
            st.caption("Character count per chunk")

def about_section():
    """About Section"""
    st.markdown('<h2 class="sub-header">ℹ️ About PDF RAG Q&A System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 What is this system?
    
    This is a **Retrieval-Augmented Generation (RAG)** system that allows you to:
    - Upload PDF documents
    - Ask questions in natural language
    - Get AI-powered answers with page citations
    - View relevant PDF pages with visual content
    
    ### 🔧 How it works:
    
    1. **📄 PDF Processing**: Extracts text and converts pages to images
    2. **📝 Text Chunking**: Breaks document into manageable pieces
    3. **🧠 Vector Embeddings**: Converts text to numerical representations
    4. **🔍 Semantic Search**: Finds relevant content based on meaning
    5. **🤖 AI Answers**: Generates responses using GPT-4o Mini
    6. **🖼️ Visual Pages**: Shows original PDF pages with charts and formatting
    
    ### 🛠️ Technology Stack:
    
    - **Frontend**: Streamlit
    - **AI Model**: OpenAI GPT-4o Mini
    - **Embeddings**: OpenAI text-embedding-3-small
    - **Vector Database**: ChromaDB
    - **PDF Processing**: PyPDF2, pdf2image
    - **Text Processing**: LangChain
    
    ### 🔑 Requirements:
    
    - OpenAI API key (get from https://platform.openai.com/api-keys)
    - PDF documents (any size, multiple pages supported)
    - Internet connection for AI processing
    
    ### 📊 Features:
    
    - ✅ **Multi-page PDF support**
    - ✅ **Visual page retrieval**
    - ✅ **Semantic question answering**
    - ✅ **Page citations and references**
    - ✅ **Chat history tracking**
    - ✅ **Document analytics**
    - ✅ **Interactive web interface**
    
    ### 🎉 Perfect for:
    
    - 📈 **Business reports analysis**
    - 📋 **Research paper exploration**
    - 📊 **Financial document review**
    - 📝 **Technical manual queries**
    - 🔍 **Legal document search**
    """)
    
    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit and OpenAI**")

# =============================================================================
# REQUIREMENTS.TXT CONTENT
# =============================================================================

def show_requirements():
    """Show requirements.txt content"""
    requirements = """
streamlit==1.28.0
streamlit-option-menu==0.3.6
langchain==0.0.354
langchain-openai==0.0.2
langchain-community==0.0.10
chromadb==0.4.18
pypdf2==3.0.1
pdf2image==1.16.3
openai==1.3.7
python-dotenv==1.0.0
matplotlib==3.7.2
pillow==10.0.1
pandas==2.0.3
"""
    return requirements

# =============================================================================
# RUN THE APP
# =============================================================================

if __name__ == "__main__":
    main()
