import streamlit as st
from langchain.memory import ConversationBufferMemory
from conversationbot_Janism import start_chat
import os

# --- Page configuration ---
st.set_page_config(
    page_title="Bhatambar Chatbot",
    page_icon="🤖",
    layout="wide")

def on_radio_change():
    st.write("Changed to:", st.session_state.choice)

api_key = st.secrets["GROQ_API_KEY"]
st.session_state.api_key = api_key
os.environ["GROQ_API_KEY"] = api_key

if "GROQ_API_KEY" not in os.environ:
    st.warning("Please enter your Groq API Key in the sidebar to start chatting.")
    st.stop()

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# --- Main layout ---
st.markdown("<h2 style='text-align: left;'>🤖 Bhatambar, Upnishad, Geeta and jain Aagams Chatbot</h2>", unsafe_allow_html=True)
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
        bot_reply = start_chat(user_input, session_id, st.session_state.api_key,'groq')
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





