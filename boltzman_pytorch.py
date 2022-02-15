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
		p_h = torch.sigmoid(torch.matmul(v,self.W.t()) + self.h_bias)
		sample_h = self.sample_from_vec(p_h)
		return sample_h
	

	def h_to_v(self,h):
		"""
			Reconstruct input from hidden layer
		"""
		p_v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
		sample_v = self.sample_from_vec(p_v)
		return sample_v
		
   	
	def forward(self,v):
		"""
			Forward propagation of the model
		"""
		h1 = self.v_to_h(v)
		
		hidden_probability = h1

		# k means how many times do we have to sample
		for _ in range(self.k):
			reconstructed_input = self.h_to_v(hidden_probability)
			hidden_probability = self.v_to_h(reconstructed_input)
		
		return v,reconstructed_input
	
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

	# load all the dataset
	handwritten_data = torchvision.datasets.ImageFolder("./data/handwrittendataset/Train/", transform)
	dataloader = torch.utils.data.DataLoader(dataset=handwritten_data, batch_size = 128, shuffle = True)


	rbm = RBM(k=1)
	loss_boltzman = np.array([])

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


	torch.save(rbm.state_dict(), "saved_models/rbm_nepali_characters.pth")



