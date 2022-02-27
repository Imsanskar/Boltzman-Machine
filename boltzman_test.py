
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import os


class RBM(nn.Module):
   def __init__(self,
               n_vis=784,
               n_hin=500,
               k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
   def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
   def get_hidden_distribution(self, v: torch.Tensor):
#         v = v.copy_(v).view(v.size()[0],784)
        
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
            
        return h_


if __name__ ==  "__main__":
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data',
        train=True,
        download = True,
        transform = transforms.Compose(
            [transforms.ToTensor()])
        ),
        batch_size=batch_size
    )

    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data',
        train=False,
        transform=transforms.Compose(
        [transforms.ToTensor()])
        ),
        batch_size=batch_size)

    rbm_mnist = RBM(k=1)
    train_op = optim.SGD(rbm_mnist.parameters(),0.1)


    path_mnist = "./saved_models/rbm_mnist_numbers.pth"
    if os.path.exists(path_mnist):
        rbm_mnist.load_state_dict(torch.load(path_mnist))
        print("RBM model found")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Grayscale()
    ])

    for i, (data, target) in enumerate(test_loader):
        plt.subplot(5, 5, i + 1)
        
        image_width = 28
        image_height = 28
        image_transform = transforms.Compose([
            transforms.Resize((image_width, image_height))
        ])
        image_1 = data[0].view((28, 28))
        image_2 = rbm_mnist.forward(data[0].view(784))[1].view((28, 28))
        image_3 = image_transform(torch.reshape(
            rbm_mnist.get_hidden_distribution(data[0].view(784)), (25, 20)).unsqueeze(0))[0]
    #     print(f"{image_1.shape}, {image_2.shape}")
        image_show = torch.cat(
            (image_1, image_2, image_3),
            1
        )
        plt.imshow(image_show.detach().cpu().numpy(), cmap=plt.cm.RdBu,
                interpolation='nearest', vmin=-2.5, vmax=2.5)
        plt.axis('off')
        
        if i >= 24:
            break

    plt.savefig("output.png")
    plt.show()


