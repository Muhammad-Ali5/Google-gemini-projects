import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import speech_recognition as sr
import os

# Page config
st.set_page_config(page_title="Voice Chatbot", page_icon="🎤")

# Title
st.title("🎤 Voice-Enabled Chatbot")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key", type="password")
    
    st.markdown("---")
    st.header("Voice Settings")
    st.info("Click the 🎤 Voice Input button to speak!")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""

# Initialize the model
@st.cache_resource
def init_model(api_key):
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            streaming=True
        )
    return None

# Voice input function
def voice_to_text():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=1)
            # Listen for audio
            audio = r.listen(source, timeout=10, phrase_time_limit=30)
            
        st.info("🔄 Processing your speech...")
        # Convert speech to text
        text = r.recognize_google(audio)
        st.success(f"✅ You said: '{text}'")
        return text
    except sr.WaitTimeoutError:
        st.error("❌ No speech detected. Please try again.")
        return ""
    except sr.UnknownValueError:
        st.error("❌ Could not understand the speech. Please try again.")
        return ""
    except sr.RequestError as e:
        st.error(f"❌ Error with speech recognition: {e}")
        return ""
    except Exception as e:
        st.error(f"❌ Microphone error: {e}")
        return ""

# Get response from model
def get_response(user_input, llm):
    try:
        # Get chat history from memory
        history = st.session_state.memory.chat_memory.messages
        
        # Create messages list
        messages = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                messages.append(HumanMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(AIMessage(content=msg.content))
        
        # Add current user message
        messages.append(HumanMessage(content=user_input))
        
        # Get response
        response = llm.invoke(messages)
        
        # Save to memory
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(response.content)
        
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# Process user input (text or voice)
def process_user_input(user_input):
    if user_input.strip():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(user_input, llm)
            
            # Display response with streaming effect
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate streaming
            for char in response:
                full_response += char
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear voice input
        st.session_state.voice_input = ""
        st.rerun()

# Main chat interface
if api_key:
    llm = init_model(api_key)
    
    # Voice input button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🎤 Voice Input", help="Click to speak"):
            voice_text = voice_to_text()
            if voice_text:
                st.session_state.voice_input = voice_text
    
    # Display voice input if available
    if st.session_state.voice_input:
        st.info(f"Voice Input: {st.session_state.voice_input}")
        if st.button("✅ Send Voice Message"):
            process_user_input(st.session_state.voice_input)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Text input
    if prompt := st.chat_input("Type your message or use voice input above..."):
        process_user_input(prompt)

else:
    st.warning("Please enter your Google API Key in the sidebar to start chatting!")
    st.info("""
    **Setup Instructions:**
    1. Get API Key from [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Install required packages:
        ```bash
       pip install streamlit langchain langchain-google-genai SpeechRecognition pyaudio
       ```
    3. Make sure your microphone is working
    """)

# Footer
st.markdown("---")
st.markdown("🎤 **Voice Features:** Click 'Voice Input' → Speak → Send")
st.markdown("💬 **Text Features:** Type in the chat box below")
st.markdown("Built with ❤️ using Streamlit, LangChain & Speech Recognition")