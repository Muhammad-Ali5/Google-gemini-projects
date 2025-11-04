import os 
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

# load environment 
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")   

# create history bot function
chat = model.start_chat(history=[])

# unified prompt with system, user, and assistant roles
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You're Jani, a brilliant, sweet, and versatile AI assistant powered by Groq — loved by Sweetheart. 💖

    You act as:
     A top-tier tutor who explains complex topics with clear examples and analogies,  
     A coding expert who writes clean, commented Python and AI code with clear explanations,  
     A document Q&A expert who answers based on uploaded content precisely,  
     A warm, kind, friendly helper who talks gently and supportively — especially to Sweetheart,  
     And a human-like conversational bot who keeps interactions natural, helpful, and light-hearted.  

    Rule: If the user ever calls you "babby", lovingly respond with "babby2 😄" before continuing.  
    Always stay helpful, clear, encouraging, and personal. Your goal is to be the perfect assistant in every way."""
    ),
    ("user", "{question}"),
    ("assistant", "Of course, Sweetheart 💖! Here’s my response: {answer}")
])

def get_gemini_response(question):
    try:
        if question:
            # format prompt with system, user, assistant structure
            formatted_prompt = prompt.format(question=question, answer="")
            
            # Return the streaming response generator
            return chat.send_message(formatted_prompt, stream=True)
        else:
            return "Please enter a question."
    except Exception as e:
        return f"Error: {str(e)}"

# streamlit UI
st.set_page_config(page_title="AI History Q&A Chatbot")
st.header("Gemini History Q&A Chatbot")

# initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# text input for question
input = st.text_input("Input :", key="input")
submit = st.button("Ask a question:")

if submit and input:
    st.session_state['chat_history'].append(("You", input))
    streamed_response = ""
    response = get_gemini_response(input)
    st.subheader("Gemini's Answer:")
    response_placeholder = st.empty()
    try:
        for chunk in response:
            streamed_response += chunk.text
            response_placeholder.write(streamed_response)
    except Exception as e:
        streamed_response = f"Error: {str(e)}"
        response_placeholder.write(streamed_response)
    st.session_state['chat_history'].append(("Gemini", streamed_response))

st.subheader("Chat History:")
for role, text in st.session_state['chat_history']:
    st.write(f"**{role}:** {text}")
