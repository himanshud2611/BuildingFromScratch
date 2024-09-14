import numpy as np
from collections import defaultdict
import re

class VectorDatabase:
    def __init__(self, embedding_size=5, random_seed=42):
        self.embedding_size = embedding_size
        np.random.seed(random_seed)  # Set a seed for reproducibility
        self.word_embeddings = defaultdict(lambda: np.random.randn(embedding_size))
        self.data = {}

        # Initialize W and b once (make them deterministic)
        self.W = np.random.randn(self.embedding_size, self.embedding_size)
        self.b = np.random.randn(self.embedding_size)

    def preprocess_text(self, text):
        # Convert to lowercase and split into words
        return re.findall(r'\w+', text.lower())

    def text_to_vector(self, text):
        words = self.preprocess_text(text)
        word_vectors = [self.word_embeddings[word] for word in words]
        return np.mean(word_vectors, axis=0)

    def encoder(self, vector):
        # Apply ReLU and a linear transformation using pre-initialized W and b
        return np.maximum(0, np.dot(self.W, vector) + self.b)

    def add_entry(self, text):
        initial_vector = self.text_to_vector(text)
        encoded_vector = self.encoder(initial_vector)
        self.data[text] = encoded_vector

    def query(self, text):
        query_vector = self.encoder(self.text_to_vector(text))
        similarities = {}
        for entry, vector in self.data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities[entry] = similarity
        return max(similarities, key=similarities.get)



# Usage example
db = VectorDatabase()

# Add entries
db.add_entry("OpenAI recently released o1 model?")
db.add_entry("o1 is one of the part of strawberry release by OpenAI")
db.add_entry("Language Model industry is changing daily")

# Query
query = "describe a bit about o1?"
result = db.query(query)
print(f"Query: '{query}', Most similar entry: '{result}'")
