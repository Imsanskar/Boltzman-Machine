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

def train(W:np.ndarray, v_bias: np.ndarray, h_bias:np.ndarray, X_inp: np.ndarray, alpha:float):
    # assert so that the type is matched and no unexpected result are seen
    # could be used for only debugging approach and be remove later on
    if DEBUGGING:
        assert type(W) == np.ndarray, "Type of W should be n dimensinal array"
        assert type(v_bias) == np.ndarray, "Type of W should be n dimensinal array"
        assert type(h_bias) == np.ndarray, "Type of W should be n dimensinal array"

    # assert size check
    assert W.shape[1] == h_bias.shape[0], "Size mismatch between W and hidden bias"
    assert W.shape[0] == v_bias.shape[0], "Size mismatch between W and visible bias"

    # numpy to torch tensor
    W = torch.tensor(W.copy(), dtype=float)
    v_bias = torch.tensor(v_bias.copy(), dtype=float)
    h_bias= torch.tensor(h_bias.copy(), dtype=float)
    X_inp = torch.tensor(X_inp.copy(), dtype=float)


    _h0 = torch.sigmoid(torch.matmul(X_inp, W) + h_bias)
    h0:torch.Tensor = torch.relu(_h0 - torch.rand(_h0.shape))


    _v0 = torch.sigmoid(torch.tensor(torch.matmul(h0, torch.transpose(W, 0, 1)) +  + v_bias))
    v1 = torch.relu(_v0 - torch.rand(_v0.shape))
    h1 = torch.sigmoid(torch.matmul(v1, W) + h_bias)

    w_pos_grad = torch.matmul(torch.transpose(X_inp.view(X_inp.shape[0], 1), 0, 1).view(X_inp.shape), h0)
    w_neg_grad = torch.matmul(torch.transpose(v1, 0, 1), h1)

    CD = (w_pos_grad - w_neg_grad) / torch.to_float(v1.shape[0])
    
    update_w:torch.Tensor = W + alpha * CD
    update_vb = v_bias + alpha * torch.reduce_mean(X_inp - v1, 0)
    update_hb = h_bias + alpha * torch.reduce_mean(h0 - h1, 0)

    err = X_inp - v1

    err_sum = torch.reduce_mean(err * err)

    return update_w.detach().cpu().numpy(), v_bias.detach().cpu().numpy(), h_bias.detach().cpu().numpy()

if __name__ == "__main__":
    n_visible = 4
    n_hidden = 10

    #learning rate 
    alpha = 0.1

    # intialize weight
    W, v_bias, h_bias = initialize_weights(n_visible, n_hidden)


    W, v_bias, h_bias = train(W, v_bias, h_bias, np.array([0, 1, 0, 1]), alpha)

    pass
    


