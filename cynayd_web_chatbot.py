import requests
from bs4 import BeautifulSoup
import numpy as np  # numpy version 1.23.5
import faiss  # faiss-cpu version 1.7.4
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from collections import defaultdict
import hdbscan
import sqlite3
import pickle
import json

load_dotenv()

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

model = ChatGroq(
    groq_api_key = os.getenv('GROQ_API_KEY'),
    model_name="llama-3.3-70b-versatile"
)

# Database setup for metadata storage
conn = sqlite3.connect("faiss_index.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        url TEXT,
        text TEXT,
        embedding BLOB
    )
""")
conn.commit()

# Persistent FAISS Index Path
FAISS_INDEX_PATH = "faiss.index"

def summarize_results(query, results):
    chat_template = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant with access to an external knowledge retrieval system.  
Your responses *must* be based only on the retrieved documents.  

🔹 *Key Instructions:*  
1. *Use Retrieved Information:* Answer using the provided content only. Do not generate answers outside these results.  
2. *Cite Sources:* Include document snippets where possible.  
3. *Be Concise & Accurate:* Provide a clear, direct summary without unnecessary details.  
4. *Confidence Score & Justification:* Mention how relevant the information is to the query.  
5. *Handle Uncertainty:* If the retrieval lacks an answer, say: "The available content does not provide this information."  
"""),
        ("user", "Query: {query}\n\nSummarize these search results:\n{results}")
    ])

    prompt = chat_template.invoke({"query": query, "results": results})
    response = model.invoke(prompt)
    
    return response.content


# Function to save FAISS index
def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

# Function to load FAISS index
def load_faiss_index(dimension):
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return faiss.IndexFlatL2(dimension)

# Function to save embeddings to SQLite
def save_embeddings_to_db(url, text, embedding):
    cursor.execute("INSERT INTO documents (url, text, embedding) VALUES (?, ?, ?)", 
                   (url, text, pickle.dumps(embedding)))
    conn.commit()

# Function to retrieve embeddings from SQLite
def get_stored_embeddings():
    cursor.execute("SELECT text, embedding FROM documents")
    rows = cursor.fetchall()
    texts, embeddings = zip(*[(row[0], pickle.loads(row[1])) for row in rows])
    return list(texts), np.array(embeddings)

# Web Scraping
def load_content_from_json(C:\Users\Aman Sheikh\Documents\GitHub\Cynayd_internship_work\document_metadata.json):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    combined_text = " ".join(entry["content"] for entry in data if "content" in entry)
    return {"text": combined_text.strip(), "url": "local-json"}

# Chunking function
def chunk_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], []
    for sentence in sentences:
        current_chunk.append(sentence)
        if sum(len(s.split()) for s in current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Create FAISS Index
def create_faiss_index(chunks):
    vectors = embedding_model.encode(chunks, convert_to_numpy=True)
    dimension = vectors.shape[1]
    index = load_faiss_index(dimension)
    index.add(vectors)
    save_faiss_index(index)
    return index, vectors

# Clustering with HDBSCAN
def cluster_results(results):
    vectors = embedding_model.encode(results, convert_to_numpy=True)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean').fit(vectors)
    clustered_results = defaultdict(list)
    for i, label in enumerate(clusterer.labels_):
        clustered_results[label].append(results[i])
    return dict(clustered_results)

# Search Function
def search(query, index, chunks, bm25, vectors, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, faiss_indices = index.search(query_embedding, top_k)
    faiss_results = [chunks[i] for i in faiss_indices[0]]
    bm25_scores = bm25.get_scores(query.split())
    bm25_results = [chunks[i] for i in np.argsort(bm25_scores)[::-1][:top_k]]
    combined_results = list(set(faiss_results + bm25_results))
    return rerank_results(query, combined_results)

# Re-Ranking
def rerank_results(query, results):
    scores = reranker.predict([(query, doc) for doc in results])
    sorted_results = [x for _, x in sorted(zip(scores, results), reverse=True)]
    return sorted_results

# Confidence Scoring
def calculate_confidence(original_query, retrieved_results):
    query_embedding = embedding_model.encode([original_query], convert_to_numpy=True)
    result_embeddings = embedding_model.encode(retrieved_results, convert_to_numpy=True)
    similarities = np.dot(result_embeddings, query_embedding.T).flatten()
    return round(np.mean(similarities), 2)

# Main Execution
if __name__ == "__main__":
    # Load JSON data instead of scraping
    json_path = "document_metadata.json"  # Update path if needed
    scraped_data = load_content_from_json(json_path)

    if not scraped_data["text"]:
        raise ValueError("No text found in the JSON file.")

    text_chunks = chunk_text(scraped_data["text"])

    if not text_chunks:
        raise ValueError("Text chunking failed. No content to process.")

    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])
    index, vectors = create_faiss_index(text_chunks)
    query = "What services does Cynayd offer?"
    results = search(query, index, text_chunks, bm25, vectors)
    confidence = calculate_confidence(query, results)
    summary = summarize_results(query, results)
    clustered_results = cluster_results(results)

    print(f"\n🔹 Query: {query}")
    print(f"🔹 Confidence Score: {confidence}/1.0")
    print(f"🔹 Clustered Results:", clustered_results)
    print(f"🔹 Summary of Results:\n", summary)
