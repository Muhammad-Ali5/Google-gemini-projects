import os 
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-pro")

def get_gemini_responce(question):
    response = model.generate_content(question)
    return response.text

# Now we Create Streamlit UI app
st.set_page_config(page_title="AI Q&A Chatbot", page_icon=":robot_face:")
st.header("Chatbot with Gemini LLM Application")

input = st.text_input("Input : ", key = "input")
submit = st.button("Ask any question")

# Submit Button is Clicked 
if submit:
    response = get_gemini_responce(input)
    st.subheader("The Output is : ")
    st.write(response)