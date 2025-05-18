from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is awesome.",
    "Artificial intelligence is the future.",
    "I love pizza."
]

embeddings = model.encode(sentences)

for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embeddings[i][:5]}...")
