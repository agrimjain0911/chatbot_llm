import streamlit as st
from langchain.memory import ConversationBufferMemory
from conversationbot_graph import start_chat
import os

# --- Page configuration ---
st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar: API Key ---
st.sidebar.header("API Configuration")
def on_radio_change():
    st.write("Changed to:", st.session_state.choice)



with st.sidebar.form("key_form"):
    choice = st.radio(
        "Select view",
        ["Groq","Open AI"],
        horizontal=True
    )


    key = st.text_input(
        "Groq API Key",
        type="password",
        help="Key is stored in session memory only"
    )
    submit = st.form_submit_button("Save Key")
    if submit:
        key = key.strip()
        if choice=='Groq':
            os.environ["GROQ_API_KEY"] = key
        else:
            os.environ["OPENAI_API_KEY"] = key
        st.session_state.api_key = key
        st.sidebar.success("API key saved ✅")

if "GROQ_API_KEY" not in os.environ:
    st.warning("Please enter your Groq API Key in the sidebar to start chatting.")
    st.stop()

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# --- Main layout ---
st.markdown("<h1 style='text-align: left;'>🤖 Company Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size:16px;'>We are using <b>moonshotai/kimi-k2-instruct-0905</b> model</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Display chat history ---
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# --- Chat input ---
user_input = st.chat_input("Ask me anything...")
if user_input and user_input.strip():
    session_id = "11111"
    if "api_key" in st.session_state:
        bot_reply = start_chat(user_input, session_id, st.session_state.api_key,choice)
    else:
        bot_reply = "API Key not provided or invalid"

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display bot reply
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    # Save to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Add to memory
    if bot_reply and isinstance(bot_reply, str):
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(bot_reply)
