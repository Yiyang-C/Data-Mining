import sys
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def readInput(fp):
	data = []
	for line in open(fp, 'r', encoding='utf-8'):
		data.append(json.loads(line))
	return data

if __name__ == '__main__':
	# edge_filename = './data/non-spherical.json'
	# output_filename = './testout5.png'

	edge_filename = sys.argv[1]
	output_filename = sys.argv[2]

	data = readInput(edge_filename)
	nodes = data[0]

	neigh = NearestNeighbors(n_neighbors=50, radius=0.2)
	neigh.fit(nodes)

	L = np.zeros((len(nodes), len(nodes)))
	for node in nodes:
		k_neighbors_list = neigh.kneighbors([node], return_distance=False)[0].tolist()
		L[k_neighbors_list[0]][k_neighbors_list[0]] = len(k_neighbors_list)-1
		for i in range(1, len(k_neighbors_list)):
			L[k_neighbors_list[0]][k_neighbors_list[i]] = -1

	E = np.linalg.eig(L)

	E_val = []
	for i, val in enumerate(E[0]):
		E_val.append((i, np.real(val)))

	E_val_sorted = sorted(E_val, key=lambda x: x[1])

	for i, eig_val in E_val_sorted:
		if eig_val > 1e-5:
			embed_i = i
			break

	T = np.zeros((len(nodes), 1))
	j = 0
	for val in E[1][:,embed_i]:
		T[j][0] = np.real(val)
		j += 1

	kmeans = KMeans(n_clusters=2, random_state=0).fit(T)

	res = kmeans.labels_
	res = res.tolist()

	xaxis1, xaxis2, yaxis1, yaxis2 = [], [], [], []
	for node, val in zip(nodes, res):
		if val == 0:
			xaxis1.append(node[0])
			yaxis1.append(node[1])
		else:
			xaxis2.append(node[0])
			yaxis2.append(node[1])

	plt.scatter(xaxis1, yaxis1, c='blue') 
	plt.scatter(xaxis2, yaxis2, c='red') 
	plt.savefig(output_filename)
	# plt.show()