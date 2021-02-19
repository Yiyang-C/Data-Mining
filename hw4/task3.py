import sys
import numpy as np
import collections
from sklearn.cluster import KMeans

if __name__ == '__main__':
	# edge_filename = './data/email-Eu-core.txt'
	# output_filename = './testout3.txt'
	# k = 42

	edge_filename = sys.argv[1]
	output_filename = sys.argv[2]
	k = int(sys.argv[3])

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
			if edge[1] not in d:
				d[edge[1]] = set()
			continue
		if edge[0] in d:
			d[edge[0]].add(edge[1])
		else:
			d[edge[0]] = {edge[1]}
		if edge[1] in d:
			d[edge[1]].add(edge[0])
		else:
			d[edge[1]] = {edge[0]}

	nodes = [k for k in d.keys()]
	node_id_idx = {}
	node_idx_id = {}
	for i in range(len(nodes)):
		node_id_idx[nodes[i]] = i
		node_idx_id[i] = nodes[i]

	L = np.zeros((len(nodes), len(nodes)))

	for nid in d:
		idx = node_id_idx[nid]
		L[idx][idx] = len(d[nid])
		for i in d[nid]:
			cur_idx = node_id_idx[i]
			L[idx][cur_idx] = -1

	E = np.linalg.eig(L)

	E_val = []
	for i, val in enumerate(E[0]):
		E_val.append((i, np.real(val)))

	E_val_sorted = sorted(E_val, key=lambda x: x[1])

	original_cnt = cnt = 3 # chose #cnt of eigenvalue as node embedding
	embed_i = []
	for i, eig_val in E_val_sorted:
		if cnt == 0:
			break
		elif eig_val > 1e-5:
			embed_i.append(i)
			cnt -= 1

	T = np.zeros((len(nodes), original_cnt))
	for i,idx in enumerate(embed_i):
		j = 0
		for val in E[1][:,idx]:
			T[j][i] = np.real(val)
			j += 1

	kmeans = KMeans(n_clusters=k, random_state=0).fit(T)

	res = kmeans.labels_
	res = res.tolist()
	# print(res)

	output_file = open(output_filename, 'w')
	for i, val in enumerate(res):
		output_file.write(str(i) + ' ' + str(val))
		if i != len(res) - 1:
			output_file.write('\n')
	output_file.close()
