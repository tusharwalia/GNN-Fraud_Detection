import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random




class GraphSage(nn.Module):
	
	def __init__(self, num_classes, enc):
		super(GraphSage, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		embeds = self.enc(nodes)
		scores = self.weight.mm(embeds)
		return scores.t()

	def to_prob(self, nodes):
		pos_scores = torch.sigmoid(self.forward(nodes))
		return pos_scores

	def loss(self, nodes, labels):
		scores = self.forward(nodes)
		return self.xent(scores, labels.squeeze())


class MeanAggregator(nn.Module):
	
	def __init__(self, features, cuda=False, gcn=False):
		
		super(MeanAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample=10):
		# Local pointers to functions (speed hack)
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh,
										num_sample,
										)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs

		if self.gcn:
			samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats


class Encoder(nn.Module):


	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 num_sample=10,
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
		super(Encoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
	
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
											  self.num_sample)

		if isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = nodes

		if not self.gcn:
			if self.cuda:
				self_feats = self.features(index)
			else:
				self_feats = self.features(index)
			combined = torch.cat((self_feats, neigh_feats), dim=1)
		else:
			combined = neigh_feats
		combined = F.relu(self.weight.mm(combined.t()))
		return combined