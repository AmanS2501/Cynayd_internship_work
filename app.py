from langchain_google_genai import ChatGoogleGenerativeAI # Changed import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

# Ensure GOOGLE_API_KEY is loaded from your .env file
print("Loaded GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY")) # Optional: for verification

# Set GOOGLE_API_KEY environment variable if it's not already set externally
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") # This line is crucial if you are loading from .env

# langsmith tracking (remains the same)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# prompt template (remains the same)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("human", "Question:{question}")
    ]
)

# streamlit app (remains the same)
st.title("Langchain Chatbot with Gemini") # Changed title for clarity
input_text = st.text_input("Enter your question:")

# Gemini LLM
# Initialize ChatGoogleGenerativeAI with the desired model, e.g., "gemini-pro"
# Ensure your GOOGLE_API_KEY is set in your environment or passed directly.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25") # Changed LLM initialization
output_parser = StrOutputParser() # Remains the same
chain = prompt | llm | output_parser # Remains the same

if input_text:
    st.write(chain.invoke({"question": input_text}))