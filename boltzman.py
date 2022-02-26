import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision

DEBUGGING: bool = True
# try gpu if available
device = "cuda" if torch.cuda.is_available()  else "cpu"

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
    W = np.zeros([n_visible, n_hidden])
    b = np.zeros([n_visible])
    c = np.zeros([n_hidden])

    return W, b, c


def save_state(W, v_bias, h_bias):
    import pickle
    pickle.dump(W, open("saved_models/Weight.p","wb" ))
    pickle.dump(h_bias, open("saved_models/h.p", "wb" ))
    pickle.dump(v_bias, open("saved_models/v.p", "wb" ))


def load_from_file():
    import pickle
    W=pickle.load(open("saved_models/Weight.p", "rb" ))
    hb=pickle.load(open("saved_models/h.p", "rb" ))
    vb=pickle.load(open("saved_models/v.p", "rb" ))

    return W, hb, vb



def sample_prob(v):
    return torch.relu(torch.sign(v - torch.randn(v.size())))


def v_to_h(W:torch.Tensor, h_bias:torch.Tensor, v:torch.Tensor):
    p_h = F.sigmoid(F.linear(v.double(), W.t().double(), h_bias.double()))
    sample_h = sample_prob(p_h)
    return p_h,sample_h

def h_to_v(W, v_bias, h):
    p_v = F.sigmoid(F.linear(h, W, v_bias))
    sample_v = sample_prob(p_v)
    return p_v,sample_v


def train(W:np.ndarray, v_bias: np.ndarray, h_bias:np.ndarray, X_inp: np.ndarray, alpha:float = 0.1, gibbs_sampling:int = 2):
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
    W = torch.tensor(W.copy())
    v_bias = torch.tensor(v_bias.copy())
    h_bias= torch.tensor(h_bias.copy())
    # X_inp = torch.tensor(X_inp.copy(), dtype=float)


    # calculation of probabilities of hidden units and sample hidden activation vector
    prob_h0, sample_hk = v_to_h(W, h_bias, X_inp)

    # calculate probability of visible vector from hidden state
    # and resample to hidden vector
    # # continue this process for gibbs sampling time 
    for _ in range(gibbs_sampling):
        prob_vk, sample_vk = h_to_v(W, v_bias, sample_hk)
        prob_hk, sample_hk = v_to_h(W, h_bias, sample_vk)


    # contrastive divergence calculation
    w_pos_grad = torch.matmul(prob_hk.t().float(), X_inp).t()
    w_neg_grad = torch.matmul(prob_hk.t().float(), sample_vk.float()).t()

    CD = (w_pos_grad - w_neg_grad)
    
    # update parameters according to rules
    # refer to https://mohitd.github.io/2017/11/25/rbms.html for more details
    update_w:torch.Tensor = W + alpha * CD
    update_vb = v_bias + alpha * (X_inp - sample_vk)
    update_hb = h_bias + alpha * (prob_hk - prob_h0)

    err = X_inp - sample_vk

    err_sum = torch.mean(err * err)

    return update_w.detach().cpu().numpy(), v_bias.detach().cpu().numpy(), h_bias.detach().cpu().numpy()


def predict(W, v_bias, h_bias, X_inp, gibbs_sampling = 3):
    # assert size check
    assert W.shape[1] == h_bias.shape[0], "Size mismatch between W and hidden bias"
    assert W.shape[0] == v_bias.shape[0], "Size mismatch between W and visible bias"

    # numpy to torch tensor
    W = torch.tensor(W.copy())
    v_bias = torch.tensor(v_bias.copy())
    h_bias= torch.tensor(h_bias.copy())
    # X_inp = torch.tensor(X_inp.copy(), dtype=float)


    # calculation of probabilities of hidden units and sample hidden activation vector
    prob_h0, sample_hk = v_to_h(W, h_bias, X_inp)

    # calculate probability of visible vector from hidden state
    # and resample to hidden vector
    # # continue this process for gibbs sampling time 
    for _ in range(gibbs_sampling):
        prob_vk, sample_vk = h_to_v(W, v_bias, sample_hk)
        prob_hk, sample_hk = v_to_h(W, h_bias, sample_vk)

    return sample_vk

    
if __name__ == "__main__":
    # TODO: Implement actual algorithm using actual dataset
    n_visible = 784
    n_hidden = 500

    #learning rate 
    alpha = 0.1


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Grayscale()
    ])
    mnist_data = datasets.MNIST(
        './data',
        train=True,
        transform=transforms.Compose(
        [transforms.ToTensor()])
    )
    test_data = torchvision.datasets.ImageFolder("./data/handwrittendataset/Test/", transform)

    # intialize weight
    W, v_bias, h_bias = initialize_weights(n_visible, n_hidden)
    print(f"Total dataset size: {len(mnist_data)}")
    TRAIN = True
    if TRAIN:
        W, h_bias, v_bias = load_from_file()
        for i in range(50):
            for _, (data, target) in enumerate(mnist_data):
                data = Variable(data.view((-1 ,784)))
                sample_data = data.bernoulli()

                # update throughout the epoch
                W, v_bias, h_bias = train(W, v_bias, h_bias, data, gibbs_sampling=1)


            print(f"Epoch {i + 1} completed")
        
            save_state(W, v_bias, h_bias)

        # W, h_bias, v_bias = load_from_file()
        print(W.shape, h_bias.shape, v_bias.shape)
        plt.figure(figsize=(20, 20))

        for i, (data, target) in enumerate(mnist_data):
            plt.subplot(5, 5, i + 1)
            
            image_1 = data.view((28, 28))
            image_2 = predict(W, v_bias, h_bias, data[0].view(784)).view((28, 28))
            image_show = torch.cat(
                (image_1, image_2),
                1
            )
        #     plt.imshow(data[0].view(28, 28).detach().cpu().numpy(), cmap=plt.cm.RdBu,
        #                interpolation='nearest', vmin=-2.5, vmax=2.5)
        #     plt.subplot(10, 20, i + 1)
        #     plt.imshow(rbm.forward(data[0].view(784))[0].view((28, 28)).detach().cpu().numpy(), cmap=plt.cm.RdBu,
        #                interpolation='nearest', vmin=-2.5, vmax=2.5)
            plt.imshow(image_show.detach().cpu().numpy(), cmap=plt.cm.RdBu,
                    interpolation='nearest', vmin=-2.5, vmax=2.5)
            plt.axis('off')
            
            if i >= 24:
                break
        plt.savefig("output.png")