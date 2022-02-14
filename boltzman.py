import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

DEBUGGING: bool = True
# try gpu if available
device = "cuda" if torch.cuda.is_available()  else "cpu"


# TODO: Forward and backward propagation
# TODO: Dataset and dataset laoder, if it is in text may need to tokenize it to one hot vector
# TODO: Test it on a validation set, search it or generate it yourself
# TODO: Presentation

def initialize_weights(n_visible: int, n_hidden: int):
    """
        Initialize weight matrices of appropriate size 
        n_visible: no of visible nodes
        n_hidden: no of hidden nodes

        Returns:
            W-> Weight matrix of the model
            b-> bias of the visible unit
            c-> bias of hidden unit
    """
    W = np.zeros([n_visible, n_hidden], dtype=np.float64)
    b = np.zeros([n_visible], dtype=np.float64)
    c = np.zeros([n_hidden], dtype=np.float64)

    return W, b, c

def forward(W:np.ndarray, v_bias: np.ndarray, h_bias:np.ndarray, X_inp: np.ndarray):
    # assert so that the type is matched and no unexpected result are seen
    # could be used for only debugging approach and be remove later on
    if DEBUGGING:
        assert type(W) == np.ndarray, "Type of W should be n dimensinal array"
        assert type(v_bias) == np.ndarray, "Type of W should be n dimensinal array"
        assert type(h_bias) == np.ndarray, "Type of W should be n dimensinal array"

    # assert size check
    assert W.shape[1] == h_bias.shape[0], "Size mismatch between W and hidden bias"
    assert W.shape[0] == v_bias.shape[0], "Size mismatch between W and visible bias"

    _h0 = torch.sigmoid(torch.tensor(np.matmul(X_inp, W) + h_bias))
    h0 = nn.ReLU(_h0 - torch.rand(_h0.shape))


    _v0 = nn.Sigmoid(torch.tensor(np.matmul(h0, np.transpose(W)) + v_bias))
    v1 = nn.ReLU(_h0 - np.random.rand(_v0.shape))
    h1 = nn.Sigmoid(np.matmul(v1, W) + h_bias)

    return h1
    

if __name__ == "__main__":
    n_visible = 4
    n_hidden = 10

    #learning rate 
    alpha = 0.1


    W, v_bias, h_bias = initialize_weights(n_visible, n_hidden)

    forward(W, v_bias, h_bias, np.array([0, 1, 0, 1]))


    


