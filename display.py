import streamlit as st
import sys
from pathlib import Path
import logging
import time
import json
from typing import List, Tuple, Dict, Any

# Import your RAG system components
try:
    from .front import RAGConfig, RAGSystem  # Assuming 'front' is your main RAG file
except ImportError as e:
    st.error(f"Could not import RAG components: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="SRH University Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        text-colur:black;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f4e79;
    }
    
    .user-message {
        background-color: #black;
        border-left-color: #4CAF50;
    }
    
    .bot-message {
        background-color: black;
        border-left-color: #1f4e79;
    }
    
    .source-info {
        background-color: black;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system with error handling."""
    try:
        # Load configuration
        config = RAGConfig.from_file("config.json")
        
        # Create default config if doesn't exist
        if not Path("config.json").exists():
            config.save_to_file("config.json")
        
        # Initialize RAG system
        rag_system = RAGSystem(config)
        
        # Load dataset
        dataset_file = "university.txt"
        if not Path(dataset_file).exists():
            st.error(f"Dataset file '{dataset_file}' not found! Please upload your university data.")
            return None, None
        
        # Build database
        with st.spinner("Loading and processing university data..."):
            content = rag_system.load_dataset(dataset_file)
            rag_system.build_database(content)
        
        st.success(f"✅ RAG system initialized with {len(rag_system.document_chunks)} document chunks")
        return rag_system, config
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        logger.error(f"RAG initialization error: {e}")
        return None, None

def format_confidence_indicator(confidence: float) -> str:
    """Format confidence level with appropriate styling."""
    if confidence > 0.8:
        return f'<span class="confidence-high">🔥 High ({confidence:.1%})</span>'
    elif confidence > 0.6:
        return f'<span class="confidence-medium">⭐ Medium ({confidence:.1%})</span>'
    else:
        return f'<span class="confidence-low">📝 Low ({confidence:.1%})</span>'


def display_sources(retrieved_knowledge: List[Tuple[str, float, Dict]]):
    """Display source information in an expandable section."""
    if not retrieved_knowledge:
        return
    
    with st.expander(f"📚 View {len(retrieved_knowledge)} Source(s)", expanded=False):
        for i, (chunk, similarity, metadata) in enumerate(retrieved_knowledge, 1):
            section = metadata.get('section', 'Main Document')
            word_count = metadata.get('word_count', len(chunk.split()))
            
            confidence_html = format_confidence_indicator(similarity)
            
            st.markdown(f"""
            <div class="source-info">
                <strong>Source {i}</strong> | Section: {section} | {confidence_html} | Words: {word_count}
                <br><br>
                <em>"{chunk[:200]}{'...' if len(chunk) > 200 else ''}"</em>
            </div>
            """, unsafe_allow_html=True)

def generate_streamlit_response(rag_system, query: str) -> Tuple[str, List, float]:
    """Generate response using the RAG system for Streamlit."""
    try:
        # Retrieve relevant knowledge
        retrieved_knowledge = rag_system.retrieve(query)
        
        if not retrieved_knowledge:
            return "I couldn't find relevant information in the SRH University database for your question. Please try rephrasing or ask about topics covered in our documentation.", [], 0.0
        
        # Calculate average confidence
        total_confidence = sum(similarity for _, similarity, _ in retrieved_knowledge)
        avg_confidence = total_confidence / len(retrieved_knowledge)
        
        # Prepare context for the LLM
        context_parts = []
        for i, (chunk, similarity, metadata) in enumerate(retrieved_knowledge):
            section_info = f" (Section: {metadata.get('section', 'Main')})" if metadata.get('section') else ""
            confidence_indicator = "🔥" if similarity > 0.8 else "⭐" if similarity > 0.6 else "📝"
            context_parts.append(f"{confidence_indicator} Source {i+1}{section_info} (Confidence: {similarity:.2f}):\n{chunk}")
        
        context = '\n\n'.join(context_parts)
        
        # Enhanced instruction prompt
        instruction_prompt = f'''You are an expert academic advisor for SRH University. Provide accurate, helpful information about SRH University's courses, programs, and services based ONLY on the provided context.

CONTEXT INFORMATION:
Average Confidence Level: {avg_confidence:.2f}
Number of Sources: {len(retrieved_knowledge)}

CURRENT CONTEXT:
{context}

STRICT GUIDELINES:
1. ACCURACY FIRST: Only provide information explicitly stated in the context above
2. SRH FOCUS: All responses must be about SRH University only
3. NO GUESSING: If information is not in the context, clearly state "This information is not available in my current knowledge base"
4. STRUCTURED RESPONSES: Organize information clearly with bullet points when appropriate
5. CONFIDENCE INDICATORS: When confidence is low (<0.5), mention "Based on available information..."
6. PRACTICAL FOCUS: Provide actionable information for students
7. FEES: Report semester fees as stated, don't calculate annual amounts
8. Be professional yet friendly and approachable

CURRENT STUDENT QUESTION: {query}

Provide a comprehensive answer based solely on the context above.'''

        # Generate response using ollama
        import ollama
        
        response = ollama.chat(
            model=rag_system.config.language_model,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ]
        )
        
        return response['message']['content'], retrieved_knowledge, avg_confidence
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Sorry, I encountered an error while processing your question: {e}", [], 0.0

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 SRH University Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Welcome to SRH University! I\'m here to help with your questions about courses, programs, admissions, and more.</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.config = None
        st.session_state.messages = []
        st.session_state.system_initialized = False
    
    # Sidebar for system controls and information
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Initialize system button
        if st.button("🚀 Initialize/Reload System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                rag_system, config = initialize_rag_system()
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.config = config
                    st.session_state.system_initialized = True
                    st.rerun()
        
        # System status
        if st.session_state.system_initialized and st.session_state.rag_system:
            st.success("✅ System Ready")
            
            # System statistics
            with st.expander("📊 System Stats"):
                try:
                    stats = st.session_state.rag_system.get_statistics()
                    st.write(f"**Document Chunks:** {stats.get('chunks_count', 0)}")
                    st.write(f"**Cache Entries:** {stats.get('cache_stats', {}).get('total_entries', 0)}")
                    st.write(f"**Cache Hit Rate:** {stats.get('cache_stats', {}).get('hit_rate', 0):.1%}")
                    if 'vector_db_stats' in stats:
                        st.write(f"**Vector DB Size:** {stats['vector_db_stats'].get('document_count', 0)}")
                except:
                    st.write("Stats unavailable")
            
            # Configuration
            with st.expander("⚙️ Configuration"):
                if st.session_state.config:
                    config_dict = {
                        "Chunk Size": st.session_state.config.chunk_size,
                        "Similarity Threshold": st.session_state.config.similarity_threshold,
                        "Top K Results": st.session_state.config.top_k,
                        "Hybrid Search": st.session_state.config.use_hybrid_search,
                        "Reranking": st.session_state.config.rerank_results
                    }
                    for key, value in config_dict.items():
                        st.write(f"**{key}:** {value}")
        else:
            st.warning("⚠️ System not initialized")
            st.info("Click 'Initialize/Reload System' to start")
        
        # Clear chat button
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Help section
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **How to ask better questions:**
            - Be specific (e.g., "MBA fees" vs "cost")
            - Ask about courses, admissions, requirements
            - Mention specific programs or departments
            
            **Example questions:**
            - What are the MBA program requirements?
            - How much does the Computer Science degree cost?
            - What are the admission deadlines?
            - Tell me about campus facilities
            """)
    
    # Main chat interface
    if not st.session_state.system_initialized:
        st.warning("🔄 Please initialize the system using the sidebar to start chatting.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
        else:
            confidence_html = format_confidence_indicator(message.get("confidence", 0.0))
            st.markdown(f'''
            <div class="chat-message bot-message">
                <strong>🤖 SRH Advisor {confidence_html}:</strong><br>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])
    
    # Chat input
    user_query = st.chat_input("Ask me anything about SRH University...")
    
    if user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Generate response
        with st.spinner("🔍 Searching university database..."):
            try:
                response, sources, confidence = generate_streamlit_response(
                    st.session_state.rag_system, 
                    user_query
                )
                
                # Add bot response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                    "confidence": confidence
                })
                
                # Rerun to display new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "SRH University RAG Chatbot | Powered by Advanced AI | "
        f"Session: {len(st.session_state.messages)} messages"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()