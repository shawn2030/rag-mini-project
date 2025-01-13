from embeddings import generate_embeddings
import faiss
from dataset import DOCUMENTS
from transformers import pipeline


# Generate embeddings for documents
document_embeddings = generate_embeddings(DOCUMENTS)

# Create a FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance metric
index.add(document_embeddings)       # Add document embeddings to the index

# Example query
# query = "How do machines learn from data?"
query = "What is natural language processing?"

# Generate query embedding
query_embedding = generate_embeddings([query])

# Search for the top 3 nearest documents
k = 3
distances, indices = index.search(query_embedding, k)

# Display results
print("Query:", query)
print("\nTop documents:")
for idx in indices[0]:
    print(DOCUMENTS[idx])



# Load a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Concatenate retrieved documents
context = " ".join([DOCUMENTS[idx] for idx in indices[0]])

# Generate response
response = generator(f"Context: {context}\nQuestion: {query}\nAnswer:", max_length=100, num_return_sequences=1)
print("\nGenerated Response:", response[0]["generated_text"])
