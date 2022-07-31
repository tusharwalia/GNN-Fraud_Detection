import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


from operator import itemgetter
import math



class InterAgg(nn.Module):

	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, intraggs,
				 inter='GNN', step_size=0.02, cuda=True):
		
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.step_size = step_size
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda

		# RL condition flag
		self.RL = True

		# number of batches for current epoch, assigned during training
		self.batch_num = 0

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# the activation function used by attention mechanism
		self.leakyrelu = nn.LeakyReLU(0.2)

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# weight parameter for each relation used by FD-Weight
		self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, 3))
		init.xavier_uniform_(self.alpha)

		# parameters used by attention layer
		self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
		init.xavier_uniform_(self.a)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		
		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
								 set.union(*to_neighs[2], set(nodes)))

		# calculate label-aware scores
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
		batch_scores = self.label_clf(batch_features)
		id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

		# the label-aware scores for current batch of nodes
		center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, r1_list, center_scores, r1_scores, r1_sample_num_list)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, r2_list, center_scores, r2_scores, r2_sample_num_list)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, r3_list, center_scores, r3_scores, r3_sample_num_list)

		# concat the intra-aggregated embeddings from each relation
		neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# inter-relation aggregation steps
	
		if self.inter == 'Att':
			# 1) FD-Att Inter-relation Aggregator
			combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats, self.embed_dim,
												self.weight, self.a, n, self.dropout, self.training, self.cuda)
		elif self.inter == 'Weight':
			# 2) FD-Weight Inter-relation Aggregator
			combined = weight_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.alpha, n, self.cuda)
			gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
			if train_flag:
				print(f'Weights: {gem_weights}')
		elif self.inter == 'Mean':
			# 3) FD-Mean Inter-relation Aggregator
			combined = mean_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, n, self.cuda)
		elif self.inter == 'GNN':
			# 4) Inter-relation Aggregator
			combined = threshold_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.thresholds, n, self.cuda)

		# the reinforcement learning module
		if self.RL and train_flag:
			relation_scores, rewards, thresholds, stop_flag = RLModule([r1_scores, r2_scores, r3_scores],
																	   self.relation_score_log, labels, self.thresholds,
																	   self.batch_num, self.step_size)
			self.thresholds = thresholds
			self.RL = stop_flag
			self.relation_score_log.append(relation_scores)
			self.thresholds_log.append(self.thresholds)

		return combined, center_scores


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, cuda=False):
		
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim

	def forward(self, nodes, to_neighs_list, batch_scores, neigh_scores, sample_list):
		
		# filer neighbors under given relation
		samp_neighs, samp_scores = filter_neighs_ada_threshold(batch_scores, neigh_scores, to_neighs_list, sample_list)

		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
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
		to_feats = F.relu(to_feats)
		return to_feats, samp_scores


def RLModule(scores, scores_log, labels, thresholds, batch_num, step_size):


	relation_scores = []
	stop_flag = True

	# only compute the average neighbor distances for positive nodes
	pos_index = (labels == 1).nonzero().tolist()
	pos_index = [i[0] for i in pos_index]

	# compute average neighbor distances for each relation
	for score in scores:
		pos_scores = itemgetter(*pos_index)(score)
		neigh_count = sum([1 if isinstance(i, float) else len(i) for i in pos_scores])
		pos_sum = [i if isinstance(i, float) else sum(i) for i in pos_scores]
		relation_scores.append(sum(pos_sum) / neigh_count)

	if len(scores_log) % batch_num != 0 or len(scores_log) < 2 * batch_num:
		# do not call RL module within the epoch or within the first two epochs
		rewards = [0, 0, 0]
		new_thresholds = thresholds
	else:
		# update thresholds according to average scores in last epoch
	
		previous_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-2 * batch_num:-batch_num])]
		current_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-batch_num:])]

		# compute reward for each relation and update the thresholds according to reward
		
		rewards = [1 if previous_epoch_scores[i] - s >= 0 else -1 for i, s in enumerate(current_epoch_scores)]
		new_thresholds = [thresholds[i] + step_size if r == 1 else thresholds[i] - step_size for i, r in enumerate(rewards)]

		# avoid overflow
		new_thresholds = [0.999 if i > 1 else i for i in new_thresholds]
		new_thresholds = [0.001 if i < 0 else i for i in new_thresholds]

		print(f'epoch scores: {current_epoch_scores}')
		print(f'rewards: {rewards}')
		print(f'thresholds: {new_thresholds}')

	# TODO: add terminal condition

	return relation_scores, rewards, new_thresholds, stop_flag


def filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list):
	
	samp_neighs = []
	samp_scores = []
	for idx, center_score in enumerate(center_scores):
		center_score = center_scores[idx][0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score = center_score.repeat(neigh_score.size()[0], 1)
		neighs_indices = neighs_list[idx]
		num_sample = sample_list[idx]

		# compute the L1-distance of batch nodes and their neighbors
		
		score_diff = torch.abs(center_score - neigh_score).squeeze()
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
		selected_indices = sorted_indices.tolist()

		# top-p sampling according to distance ranking and thresholds
	
		if len(neigh_scores[idx]) > num_sample + 1:
			selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
			selected_scores = sorted_scores.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_scores = score_diff.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, cuda):
	

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# sum neighbor embeddings together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :]

	# sum aggregated neighbor embedding and batch node embedding
	# take the average of embedding and feed them to activation function
	combined = F.relu((center_h + aggregated) / 4.0)

	return combined


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, cuda):
	

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# compute relation weights using softmax
	w = F.softmax(alpha, dim=1)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :] * w[:, r]

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):


	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	import pdb
	pdb.set_trace()
	# compute attention weights
	combined = torch.cat((center_h.repeat(3, 1), neigh_h), dim=1)
	e = att_layer(combined.mm(a))
	attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:3 * n, :]), dim=1)
	ori_attention = F.softmax(attention, dim=1)
	attention = F.dropout(ori_attention, dropout, training=training)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add neighbor embeddings in each relation together with attention weights
	for r in range(num_relations):
		aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu((center_h + aggregated))

	# extract the attention weights
	att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

	return combined, att


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, cuda):


	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :] * threshold[r]

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined
