import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling
st.markdown(
    """
    <style>
        /* General styles */
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #1a1a1a;
        }
        .main {
            background: linear-gradient(135deg, #1e1e2f, #1a1a1a);
            color: white;
            padding: 2rem;
        }
        .stTextInput textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: #2d2d2d !important;
            color: white !important;
            border: 1px solid #5a5a5a !important;
        }
        .chat-card {
            background: #2e2e3a;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .chat-header {
            font-size: 1.3rem;
            color: #00ffcc;
            margin-bottom: 1rem;
        }
        .clear-btn {
            background-color: #e63946;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            margin: 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and subtitle
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    clear_chat = st.button("Clear Chat History", key="clear_chat")
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# LLM engine configuration
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state or clear_chat:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    st.markdown("<div class='chat-header'>Chat Messages</div>", unsafe_allow_html=True)
    for message in st.session_state.message_log:
        role_style = "color: #00ffcc" if message["role"] == "ai" else "color: #ffffff"
        st.markdown(f"<div class='chat-card' style='{role_style}'>{message['content']}</div>", unsafe_allow_html=True)

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    def generate_ai_response(prompt_chain):
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})

    def build_prompt_chain():
        prompt_sequence = [system_prompt]
        for msg in st.session_state.message_log:
            if msg["role"] == "user":
                prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "ai":
                prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
        return ChatPromptTemplate.from_messages(prompt_sequence)

    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    # Rerun to update chat display
    st.rerun()
