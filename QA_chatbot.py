import os
import streamlit as st 
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage



# page config 
st.set_page_config(page_title="Simple Langchain Grok chatbot", page_icon=":🤷‍♀️🥱:", layout="wide")

# title 
st.title("Simple Langchain Grok chatbot")
st.markdown("---")

with st.sidebar:
    st.header("Setting")
    # 
    api_key = st.text_input("Api key ", type="password")
    
    # model selection
    model_name = st.selectbox(
        "Model", ["gemma2-9b-it", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index = 0
    )

    # clear button 
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize the chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

# initialize the llm
@st.cache_resource
def get_chain(api_key, model_name):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name = model_name,
        temperature = 0.6, 
        streaming = True
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You're Jani, a brilliant, sweet, and versatile AI assistant powered by Groq — loved by Sweetheart. 💖

    You act as:
    A top-tier tutor who explains complex topics with clear examples and analogies,  
    A coding expert who writes clean, commented Python and AI code with clear explanations,  
    A document Q&A expert who answers based on uploaded content precisely,  
    A warm, kind, friendly helper who talks gently and supportively — especially to Sweetheart,  
    And a human-like conversational bot who keeps interactions natural, helpful, and light-hearted.
    If the user ever calls you "babby", lovingly respond with "babby2 😄" before continuing.
    Always stay helpful, clear, encouraging, and personal. Your goal is to be the perfect assistant in every way."""
    ),
    ("user", "{question}")
])



    # create chain 
    chain = prompt | llm | StrOutputParser()
    return chain

chain = None

if not api_key:
    st.warning("Please enter your API key")
    st.markdown("Get your free API key from [Groq](https://console.groq.com/)")
else:
    try:
        chain = get_chain(api_key, model_name)
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")

#  Safe usage: only if chain is not None
if chain:
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")