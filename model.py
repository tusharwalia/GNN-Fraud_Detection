import torch
import torch.nn as nn
from torch.nn import init




class OneLayerFD(nn.Module):
	
	def __init__(self, num_classes, inter1, lambda_1):
		super(OneLayerFD, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1

	def forward(self, nodes, labels, train_flag=True):
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		scores = torch.mm(embeds1, self.weight)
		return scores, label_scores

	def to_prob(self, nodes, labels, train_flag=True):
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
		label_prob = nn.functional.softmax(label_scores, dim=1)
		return gnn_prob, label_prob

	def loss(self, nodes, labels, train_flag=True):
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		
		label_loss = self.xent(label_scores, labels.squeeze())
		
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
	
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss
