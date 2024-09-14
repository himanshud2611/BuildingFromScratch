import numpy as np

def self_attention(X):
    # input sequence X of shape (sequence_length, embedding_dim)
    sequence_length, embedding_dim = X.shape

    # Weights for Q, K, V
    W_q = np.random.rand(3, 6)
    W_k = np.random.rand(3, 6)
    W_v = np.random.rand(3, 6)

    # Computing Q, K, V
    Q = np.dot(W_q, X)
    K = np.dot(W_k, X)
    V = np.dot(W_v, X)

    #Computing Matmul-1
    matmul_1 = np.dot(K.T, Q)
    scale_matmul_1 = matmul_1/np.sqrt(3)

    #applying softmax on above to get attention weights
    attention_weights = softmax(scale_matmul_1)

    #Computing final output to get Matmul_2
    matmul_2 = np.dot(V, attention_weights)

    return matmul_2


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims = True))
    return exp_x/np.sum(exp_x, axis=1, keepdims = True)

X = np.random.rand(6, 4)

matmul_2 = self_attention(X)

print("Input shape: ", X.shape)
print("Matmul_2 shape: ", matmul_2.shape)
print("Output: ")
print(matmul_2)