# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Install required libraries
# !pip install langchain langchain-core langchain_community langgraph langchain-huggingface transformers torch
# !pip install unstructured

# Import necessary libraries
#from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import hub
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
# Load documents from URLs
urls = ['https://cynayd.com/', "https://cynayd.com/service-web", "https://cynayd.com/why-us"]

from huggingface_hub import login
login(token="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")  # Replace with your Hugging Face token

def load_html_from_urls(urls):
    docs = []
    for url in urls:
        try:
            print(f"Fetching {url}...")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            docs.append(Document(page_content=text))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return docs

docs = load_html_from_urls(urls)
print("Documents loaded successfully!")

print("Documents loaded successfully!")

print(docs)

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings()

# Example embedding query
# vector = embeddings.embed_query("hello, world!")
# print(vector[:5])  # Print first 5 elements of the vector

# Create a Chroma vector store from the document splits
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())

# Optional: Persist the Chroma database locally
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(),
    persist_directory="./faiss_index"  # Custom directory
)

# Load the persisted database later (optional)
# vectorstore = Chroma(
#     persist_directory="./my_chroma_db",
#     embedding_function=HuggingFaceEmbeddings()
# )

#Initialize Hugging Face text generation pipeline for CPU
model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Replace with your desired model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Configure the pipeline to use CPU
text_generation_pipeline = pipeline(
    "text-generation",
    do_sample=True,
    model=model,
    device_map="auto",
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.3,  # Lower values = more deterministic
    top_k=50,  # Filters out low-probability tokens
)

# Wrap the pipeline in a LangChain HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

custom_prompt = PromptTemplate.from_template(
    """You are an intelligent assistant. Based on the context below, extract the answer to the question mentioned in the text.

Context:
{context}

Question: {question}

Only return the answer and nothing else. If unknown, say "Answer not found, I am still learning."
"""
)
# Pull a pre-defined prompt template from LangChain Hub
prompt = custom_prompt

# Define the application state using TypedDict
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define the retrieval step
def retrieve(state: State):
    # Retrieve relevant documents based on the question
    retrieved_docs = vectorstore.similarity_search(state["question"], k=2)
    for idx, doc in enumerate(retrieved_docs):
        print(f"\n[Doc {idx+1}] Preview:\n{doc.page_content[:300]}")
    
    return {"context": retrieved_docs}



def trim_to_max_tokens(text: str, tokenizer, max_tokens=1500):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)

# Define the generation step
def generate(state: State):
    
    
    # Combine the context documents into a single string
    docs_content = "\n\n".join([doc.page_content .strip()
                               for doc in state["context"]
                               if len(doc.page_content.strip()) > 30
                               ])
    
    docs_content = trim_to_max_tokens(docs_content, tokenizer)

    
# Format the prompt with the question and context
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    print("Prompt to LLM:\n", messages)
# Generate a response using the LLM
    response = llm.invoke(messages)
    return {
    "question": state["question"],
    "context": state["context"],
    "answer": response
    }






## Build the LangGraph application
#graph_builder = StateGraph(State)
#graph_builder.add_sequence([retrieve, generate])  # Add retrieve and generate steps
#graph_builder.add_edge(START, "retrieve")  # Connect START to retrieve
print("Initializing graph...")
graph_builder = StateGraph(State)
print("Adding sequence...")
graph_builder.add_sequence([retrieve, generate])
print("Adding edge...")
graph_builder.add_edge(START, "retrieve")
print("Done setting up graph.")
graph = graph_builder.compile()  # Compile the graph





# Visualize the graph (optional)
display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke the graph with a question
response = graph.invoke({"question": input("Enter your question: ")})
print(response["answer"])