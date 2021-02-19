import sys
import numpy as np
import collections

def page_rank(matrix, d=0.8, tol=1e-2, max_iter=10000, log=False):
	"""Return the PageRank of the nodes in the graph. 

	:param dict G: the graph
	:param float d: the damping factor (teleportation)
	:param flat tol: tolerance to determine algorithm convergence
	:param int max_iter: max number of iterations
	"""
    # matrix = nx.to_numpy_matrix(G).T
	out_degree = matrix.sum(axis=0)
	N = len(nodes)
	# To avoid dead ends, since out_degree is 0, replace all the values with 1/N to make it column stochastic
	weight = np.divide(matrix, out_degree, out=np.ones_like(matrix)/N, where=out_degree!=0)
	pr = np.ones(N).reshape(N, 1) * 1./N

	for it in range(max_iter):
		old_pr = pr[:]
		pr = d * weight.dot(pr) + (1-d)/N
		if log:
			print (f'old_pr: {np.asarray(old_pr).squeeze()}, pr: {np.array(pr).squeeze()}')
		err = np.absolute(pr - old_pr).sum()
		if err < tol:
			return pr

	return pr

if __name__ == '__main__':
	# edge_filename = './data/email-Eu-core.txt'
	# output_filename = './testout6.txt'

	edge_filename = sys.argv[1]
	output_filename = sys.argv[2]

	edge_file = open(edge_filename, 'r')

	edges = []
	while True:
		line = edge_file.readline().split()
		if not line:
			break
		edges.append(line)
	edge_file.close()

	d = collections.defaultdict(set)
	for edge in edges:
		if edge[0] == edge[1]:
			if edge[0] not in d:
				d[edge[0]] = set()
			continue
		if edge[0] in d:
			d[edge[0]].add(edge[1])
		else:
			d[edge[0]] = {edge[1]}
		if edge[1] not in d:
				d[edge[1]] = set()

	nodes = [k for k in d.keys()]
	node_id_idx = {}
	node_idx_id = {}
	for i in range(len(nodes)):
		node_id_idx[nodes[i]] = i
		node_idx_id[i] = nodes[i]

	A = np.zeros((len(nodes), len(nodes)))

	for nid in d:
		idx = node_id_idx[nid]
		for i in d[nid]:
			cur_idx = node_id_idx[i]
			A[cur_idx][idx] = 1

	pr = page_rank(A)
	res = pr.tolist()
	idx_pr = []
	for i, res in enumerate(res):
		idx_pr.append((i, res))

	tmp_res = sorted(idx_pr, key=lambda x:x[1], reverse=True)
	node_res = []
	for i in range(20):
		node_res.append(tmp_res[i][0])
	# print(node_res)

	output_file = open(output_filename, 'w')
	for i in range(20):
		output_file.write(str(node_res[i]))
		if i != 19:
			output_file.write('\n')
	output_file.close()