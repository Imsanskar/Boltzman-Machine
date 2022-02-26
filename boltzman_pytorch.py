import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
from torchvision.utils import make_grid , save_image

# various constants
batch_size = 256



class RBM(nn.Module):
	def __init__(self, n_visible = 784, n_hidden = 500, k = 5) -> None:
		super(RBM, self).__init__()

		self.W: torch.Tensor = nn.parameter.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
		self.v_bias:torch.Tensor = nn.parameter.Parameter(torch.zeros(n_visible))
		self.h_bias:torch.Tensor = nn.parameter.Parameter(torch.zeros(n_hidden))

		self.k = k


	def sample_from_vec(self, vec:torch.Tensor):
		return torch.relu(torch.sign(vec - Variable(torch.randn(vec.size()))))


	def v_to_h(self,v):
		"""
			Calculates the probabilities of the hidden units
		"""
		p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
		sample_h = self.sample_from_p(p_h)
		return p_h,sample_h
	

	def h_to_v(self,h):
		"""
			Reconstruct input from hidden layer
		"""
		p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
		sample_v = self.sample_from_p(p_v)
		return p_v,sample_v
		
   	
	def forward(self,v:torch.Tensor):
		"""
			Forward propagation of the model
		"""
		pre_h1,h1 = self.v_to_h(v)
		
		h_ = h1
		for _ in range(self.k):
			pre_v_,v_ = self.h_to_v(h_)
			pre_h_,h_ = self.v_to_h(v_)
		
		return v,v_


	def get_hidden_distribution(self, v: torch.Tensor):
		v = v.copy_(v).view(v.size()[0],784)
		h1 = self.v_to_h(v)
		
		return h1
	
	def free_energy(self, inp):
		"""
			Calculates energy of the state
		"""
		# matrix vector multiplication of inp and v_bias
		vbias_term = inp.mv(self.v_bias)
		wx_b = torch.matmul(inp,self.W.t()) + self.h_bias
		hidden_term = wx_b.exp().add(1).log().sum(1)
		return (-hidden_term - vbias_term).mean()


if __name__ == "__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = RBM(k = 1)
	loss_fn = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr = 0.05)
	loss_fn.to(device)
	model.to(device)
	print(f"Device: {device}")

	transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((28, 28)),
	transforms.Grayscale()
])

	handwritten_data = torchvision.datasets.ImageFolder("./data/handwrittendataset/Train/", transform)
	test_data = torchvision.datasets.ImageFolder("./data/handwrittendataset/Test/", transform)
	batch_size = 256
	dataloader = torch.utils.data.DataLoader(dataset=handwritten_data, batch_size = batch_size, shuffle = True)
	test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size = 4, shuffle = True)


	rbm = RBM(k=1)
	rbm = RBM(k = 1)
	path = "./saved_models/rbm_nepali_characters.pth"
	if os.path.exists(path):
		rbm.load_state_dict(torch.load(path))
		print("RBM model found")
	loss_boltzman = np.array([])

	if False:
		for epoch in range(10):
			loss_ = []
			for i, (data,target) in enumerate(dataloader):
				data = Variable(data.view(data.size()[0], 784))
				sample_data = data.bernoulli()
				v,v1 = rbm(sample_data)
				loss = rbm.free_energy(v) - rbm.free_energy(v1)
				loss_.append(loss.data)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if i % 100 == 0:
					print("Training loss for {} epoch: {}".format(epoch, np.mean(loss_)))
					
			loss_boltzman = np.append(loss_boltzman, np.mean(loss_))

	plt.figure(figsize=(20, 20))

	for i, (data, target) in enumerate(test_dataloader):
		plt.subplot(10, 20, i + 1)
		plt.imshow(data[0].view(28, 28).detach().cpu().numpy(), cmap=plt.cm.RdBu,
				interpolation='nearest', vmin=-2.5, vmax=2.5)
		plt.subplot(10, 20, i + 2)
		plt.imshow(rbm.forward(data[0].view(784))[0].view((28, 28)).detach().cpu().numpy(), cmap=plt.cm.RdBu,
				interpolation='nearest', vmin=-2.5, vmax=2.5)
		plt.axis('off')
		
		if i >= 99:
			break
	plt.show()

	torch.save(rbm.state_dict(), "saved_models/rbm_nepali_characters.pth")



