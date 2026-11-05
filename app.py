
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ------------------- PAGE CONFIGURATION -------------------

st.set_page_config(
    page_title="AI Study Guide",
    page_icon="🤖",
    layout="centered"
)

load_dotenv()

# API KEY CONFIGURATION
api_key = os.getenv("gem")

if not api_key:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

# ------------------- MODEL CONFIGURATION -------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0.7
)

# ------------------- MODULES -------------------------------

modules = [
    "Python",
    "SQL",
    "Power BI",
    "EDA",
    "Machine Learning",
    "Deep Learning",
    "Generative AI",
    "Agentic AI"
]

# ------------------- SESSION STATE -------------------------

if "selected_module" not in st.session_state:
    st.session_state.selected_module = None

if "conv" not in st.session_state:
    st.session_state.conv = []

if "memory" not in st.session_state:
    st.session_state.memory = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ------------------- MAIN PAGE -----------------------------

if st.session_state.selected_module is None:

    st.title("👋 :rainbow[Welcome to AI Study Guide]")

    st.write("Your personalized AI learning assistant.")

    st.write("Please select a learning module to begin.")

    selected = st.selectbox(
        "📌 Select Module",
        ["-- Select --"] + modules
    )

    if selected != "-- Select --":
        st.session_state.selected_module = selected
        st.rerun()

    st.stop()

# ------------------- MODULE PAGE ---------------------------

module = st.session_state.selected_module

st.title(f"🎯 {module} Assistant")

st.write(f"I am your dedicated assistant for **{module}**.")

st.write("How can I help you today?")

# ------------------- SYSTEM PROMPT -------------------------

if not st.session_state.memory:

    system_prompt = f"""
    You are an AI Mentor specialized ONLY in {module}.

    Rules:
    1. Answer ONLY questions related to {module}.
    2. If the question is outside {module}, respond exactly:

    Sorry, I do not know about this question.
    Please ask something related to the selected module.

    3. Be clear, structured, and educational.
    4. Use examples wherever necessary.
    5. Format answers using Markdown.
    """

    st.session_state.memory.append(
        SystemMessage(content=system_prompt)
    )

# ------------------- CHAT HISTORY --------------------------

for msg in st.session_state.conv:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------- USER QUERY ----------------------------

prompt = st.chat_input("Type your question")

if prompt:

    st.session_state.conv.append({
        "role": "user",
        "content": prompt
    })

    st.session_state.memory.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    try:

        with st.chat_message("assistant"):

            with st.spinner("AI Mentor is thinking..."):

                response = model.invoke(
                    st.session_state.memory
                )

                # Extract response text
                content = response.content

                if isinstance(content, list):

                    content = "\n".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                    )

                if not isinstance(content, str):
                    content = str(content)

                # Display clean Markdown
                st.markdown(content)

        # Save conversation
        st.session_state.conv.append({
            "role": "assistant",
            "content": content
        })

        st.session_state.memory.append(
            AIMessage(content=content)
        )

    except Exception as e:
        st.error(f"Error generating response: {e}")

# ------------------- SIDEBAR -------------------------------

with st.sidebar:

    st.title("⚙️ Study Guide Settings")

    st.write(f"**Current Module:** {module}")

    if st.button("🔄 Change Module"):

        st.session_state.selected_module = None
        st.session_state.conv = []
        st.session_state.memory = []
        st.rerun()

    if st.button("🗑️ Clear Conversation"):

        st.session_state.conv = []
        st.session_state.memory = []
        st.rerun()

# ------------------- DOWNLOAD CHAT -------------------------

if st.session_state.conv:

    chat_text = ""

    for msg in st.session_state.conv:

        role = (
            "User"
            if msg["role"] == "user"
            else "AI Mentor"
        )

        chat_text += (
            f"{role}: {msg['content']}\n\n"
        )

    st.download_button(
        "📥 Download Conversation",
        chat_text,
        file_name=f"{module}_Chat.txt",
        mime="text/plain"
    )

    st.feedback("stars", key="user_feedback")

# ------------------- FOOTER --------------------------------

st.divider()

st.caption(
    "🤖 AI Study Guide | Powered by Google Gemini and Streamlit"
)
