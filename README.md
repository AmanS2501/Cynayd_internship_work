# 🧠 Web-Powered Chatbot using Mistral 7B, LangChain & Chroma

This project builds a document-aware chatbot that leverages the **Mistral 7B Instruct model**, **LangChain**, **BeautifulSoup**, and **Chroma vector store** to fetch web content, chunk and embed it, and allow users to query it interactively via a text-generation pipeline.

---

## 🚀 Features

- 🔗 **Web Scraping**: Pulls and processes text from specified URLs.
- 🧠 **LLM Integration**: Uses [Mistral 7B Instruct v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) via HuggingFace Transformers.
- 📚 **Contextual Retrieval**: Stores processed documents in a **Chroma** vector store for efficient similarity search.
- 🤖 **Prompt Engineering**: Uses LangChain's PromptTemplate for contextual Q&A.
- 🧱 **LangGraph**: Constructs a modular pipeline using LangGraph to retrieve context and generate answers.
- 💾 **FAISS Persistence**: Saves vector data locally using Chroma for future use.

---

## 📂 Project Structure

├── chatbot_using_mistral.py # Main script
├── faiss_index/ # Persisted Chroma vector DB
├── requirements.txt # Required dependencies

yaml
Copy
Edit

---

## 🛠️ Installation

### ✅ Prerequisites
- Python 3.10 or higher
- A Hugging Face account with an access token

### 📦 Install dependencies:

```bash
pip install -r requirements.txt
Or manually install key packages:

bash
Copy
Edit
pip install langchain langchain-core langchain_community langgraph \
            langchain-huggingface huggingface_hub \
            transformers torch chromadb unstructured beautifulsoup4
🔐 Setup
Add your Hugging Face token in the script:

python
Copy
Edit
from huggingface_hub import login
login(token="your_huggingface_token_here")
🧪 How It Works
Scrape & clean: Pulls content from Cynayd URLs.

Split & embed: Chunks content and converts it into vector embeddings using HuggingFace models.

Vector store: Stores the embeddings in a Chroma (FAISS-like) vector store.

Query handling: Accepts user input and retrieves relevant documents using similarity search.

Response generation: Constructs a prompt and generates an answer using Mistral-7B.

💬 Example
bash
Copy
Edit
> Enter your question: What services does Cynayd offer?
Fetching https://cynayd.com/...
...
Documents loaded successfully!
Prompt to LLM:
[Context + Question]
Answer: Cynayd offers AI-driven web and cloud services tailored to business needs.
📸 Visualization
LangGraph is used to build and visualize the flow:

python
Copy
Edit
display(Image(graph.get_graph().draw_mermaid_png()))
📌 To Do
 Add Streamlit/Gradio UI

 Add PDF/Doc/CSV document loaders

 Switch to Ollama/Local model backend (optional)

 Add logging and exception handling

🤝 Credits
Mistral AI

LangChain

Chroma

Transformers by Hugging Face

Cynayd

📝 License
This project is licensed under the MIT License.

python
Copy
Edit

---

Let me know if you'd like to:
- Add Gradio/Streamlit interface
- Extend it with Gemini/Google search
- Turn this into a deployable API or web app

I'm happy to help with the next steps!