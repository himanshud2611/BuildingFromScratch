import numpy as np

# Function to initialize random weight and bias matrices
def initialize_parameters(dim_in, dim_out):
    W = np.random.randn(dim_out, dim_in) * 0.01  # Small random values
    b = np.random.randn(dim_out) * 0.01  # Small random bias
    return W, b

# Linear projection function
def linear_projection(embedding, W, b):
    return np.dot(W, embedding) + b

# Example embeddings
E_text = np.random.randn(768)  # Text embedding (dimension 768)
E_image = np.random.randn(1024)  # Image embedding (dimension 1024)

# Initialize parameters for text and image projections
W_text, b_text = initialize_parameters(768, 128)
W_image, b_image = initialize_parameters(1024, 128)

# Project text and image embeddings to the 128-dimensional shared space
E_text_proj = linear_projection(E_text, W_text, b_text)
E_image_proj = linear_projection(E_image, W_image, b_image)


# Function to compute dot product similarity between query token and image patches
def similarity(e_query, E_img):
    return np.dot(E_img, e_query)

# Function to compute MaxSim for a given query token across all image patches
def max_sim(e_query, E_img):
    similarities = np.dot(E_img, e_query)  # Dot product for each image patch
    return np.max(similarities)  # Max similarity across all patches

# Late Interaction Score computation
def late_interaction(E_query, E_img):
    total_similarity = 0
    for e_query in E_query:
        total_similarity += max_sim(e_query, E_img)
    return total_similarity

# E_query is an (n, d) matrix where n is the number of query tokens, d is the dimension
# E_img is an (m, d) matrix where m is the number of image patches, d is the dimension
E_query = np.random.randn(5, 128)  # 5 query tokens, 128-dimensional embeddings
E_img = np.random.randn(10, 128)  # 10 image patches, 128-dimensional embeddings


# Print the results
print("Projected Text Embedding:", E_text_proj)
print("Projected Image Embedding:", E_image_proj)

# Compute the Late Interaction score
LI_score = late_interaction(E_query, E_img)
print("Late Interaction Score:", LI_score)
