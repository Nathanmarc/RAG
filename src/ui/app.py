import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, List

# Page configuration
st.set_page_config(
    page_title="T4 GPU Consultant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 0;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 12px;
    }
    .response-box {
        border-left: 4px solid #4CAF50;
        padding: 12px;
        margin: 10px 0;
        background-color: #f1f8f4;
        border-radius: 5px;
    }
    .error-box {
        border-left: 4px solid #f44336;
        padding: 12px;
        margin: 10px 0;
        background-color: #ffebee;
        border-radius: 5px;
    }
    .source-box {
        border-left: 4px solid #2196F3;
        padding: 10px;
        margin: 5px 0;
        background-color: #e3f2fd;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .low-confidence {
        background-color: #ff9800;
    }
    .high-latency {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")

# API Configuration
api_host = st.sidebar.text_input(
    "API Host",
    value="http://localhost:8000",
    help="FastAPI server URL"
)

# Model Parameters
col1, col2 = st.sidebar.columns(2)
with col1:
    top_k = st.number_input(
        "Top K Results",
        value=5,
        min_value=1,
        max_value=20,
        help="Number of documents to retrieve"
    )

with col2:
    max_tokens = st.number_input(
        "Max Tokens",
        value=512,
        min_value=100,
        max_value=2048,
        help="Maximum response length"
    )

# Features
st.sidebar.markdown("### 🎯 Features")
show_sources = st.sidebar.checkbox("Show Source Citations", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
show_latency = st.sidebar.checkbox("Show Performance Metrics", value=True)

# History
st.sidebar.markdown("### 📋 Session History")
clear_history = st.sidebar.button("Clear Chat History")

# Main Content
st.title("🚀 T4 GPU Knowledge Base")
st.markdown("*Expert AI for NVIDIA T4 GPU Technical Questions*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if clear_history:
    st.session_state.messages = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            # Display confidence score
            if show_confidence and "confidence" in metadata:
                confidence = metadata["confidence"]
                confidence_color = "low-confidence" if confidence < 0.7 else ""
                st.markdown(
                    f'<span class="confidence-badge {confidence_color}">Confidence: {confidence:.0%}</span>',
                    unsafe_allow_html=True
                )
            
            # Display latency
            if show_latency and "latency_ms" in metadata:
                latency = metadata["latency_ms"]
                latency_color = "high-latency" if latency > 500 else ""
                st.markdown(
                    f'<span class="confidence-badge {latency_color}">⏱️ {latency}ms</span>',
                    unsafe_allow_html=True
                )
            
            # Display sources
            if show_sources and "sources" in metadata:
                st.markdown("**📚 Sources:**")
                for source in metadata["sources"]:
                    page = source.get("page", "Unknown")
                    section = source.get("section", "Unknown")
                    st.markdown(
                        f'<div class="source-box">📄 Page {page} - {section}</div>',
                        unsafe_allow_html=True
                    )

# Chat input
col1, col2 = st.columns([10, 1])

with col1:
    user_input = st.chat_input("Ask about T4 GPU capabilities...")

with col2:
    send_button = st.button("📤")

# Process user input
if user_input or send_button:
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Call API
        with st.spinner("⏳ Thinking..."):
            try:
                response = requests.post(
                    f"{api_host}/query",
                    json={"query": user_input},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer received")
                    
                    # Extract metadata
                    metadata = {
                        "confidence": result.get("confidence", 0.5),
                        "latency_ms": result.get("latency_ms", 0),
                        "sources": result.get("sources", [])
                    }
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata
                    })
                    
                    # Display response
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        
                        if show_confidence:
                            confidence = metadata["confidence"]
                            confidence_color = "low-confidence" if confidence < 0.7 else ""
                            st.markdown(
                                f'<span class="confidence-badge {confidence_color}">Confidence: {confidence:.0%}</span>',
                                unsafe_allow_html=True
                            )
                        
                        if show_latency:
                            latency = metadata["latency_ms"]
                            latency_color = "high-latency" if latency > 500 else ""
                            st.markdown(
                                f'<span class="confidence-badge {latency_color}">⏱️ {latency}ms</span>',
                                unsafe_allow_html=True
                            )
                        
                        if show_sources and metadata["sources"]:
                            st.markdown("**📚 Sources:**")
                            for source in metadata["sources"]:
                                page = source.get("page", "Unknown")
                                section = source.get("section", "Unknown")
                                st.markdown(
                                    f'<div class="source-box">📄 Page {page} - {section}</div>',
                                    unsafe_allow_html=True
                                )
                else:
                    error_msg = f"API Error: {response.status_code}"
                    st.markdown(f'<div class="error-box">❌ {error_msg}</div>', unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {error_msg}"
                    })
            
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to API. Make sure the server is running at " + api_host
                st.markdown(f'<div class="error-box">❌ {error_msg}</div>', unsafe_allow_html=True)
            
            except Exception as e:
                error_msg = str(e)
                st.markdown(f'<div class="error-box">❌ Error: {error_msg}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
### 📚 Example Queries
- "How many YOLOv8 video streams can T4 process simultaneously?"
- "What's the optimal TensorRT configuration for T4?"
- "How to quantize my model to INT8 for T4 inference?"
- "What's the power consumption at different inference loads?"
- "What are the edge deployment constraints for T4?"

### ⚙️ Keyboard Shortcuts
- `Enter` in chat input: Send message
- `Shift+Enter`: New line in chat input
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔗 Links
- [Documentation](https://github.com/yourusername/ko)
- [API Docs](http://localhost:8000/docs)
- [MLflow Dashboard](http://localhost:5000)
- [Grafana Monitoring](http://localhost:3000)
""")

st.sidebar.markdown("""
### 📝 Version
**T4 Consultant v1.0.0**  
*Last Updated: November 2025*
""")
