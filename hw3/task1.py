import sys
import json
import networkx as nx

def readInput(fp):
	data = []
	for line in open(fp, 'r', encoding='utf-8'):
		data.append(json.loads(line))
	return data

def writeOutput(fp, data):
	with open(fp, 'w', encoding='utf-8') as outfile:
		json.dump(data, outfile)


if __name__ == '__main__':
	input_file = sys.argv[1]
	gexf_output_file = sys.argv[2]
	json_output_file = sys.argv[3]

	# input_file = './Gamergate.json'
	# input_file = './toy_test/mini_mid_gamergate.json'
	# gexf_output_file = './gext_testout1.gexf'
	# json_output_file = './json_testout1.json'

	tweets = readInput(input_file)

	G = nx.DiGraph()

	for tweet in tweets:
		if 'retweeted_status' not in tweet:
			G.add_node(tweet['user']['screen_name'])
		else:
			if G.has_edge(tweet['user']['screen_name'], tweet['retweeted_status']['user']['screen_name']):
				G[tweet['user']['screen_name']][tweet['retweeted_status']['user']['screen_name']]['weight'] += 1
			else:
				G.add_edge(tweet['user']['screen_name'], tweet['retweeted_status']['user']['screen_name'], weight = 1)

	nx.write_gexf(G, gexf_output_file)
	
	n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()

	retweeted_list = sorted([(user, weight) for user, weight in G.in_degree(weight='weight')], key=lambda x: x[1], reverse=True)
	max_retweeted_user = retweeted_list[0][0]
	max_retweeted_number = retweeted_list[0][1]

	retweeter_list = sorted([(user, weight) for user, weight in G.out_degree(weight='weight')], key=lambda x: x[1], reverse=True)
	max_retweeter_user = retweeter_list[0][0]
	max_retweeter_number = retweeter_list[0][1]

	res = {}
	res['n_nodes'] = n_nodes
	res['n_edges'] = n_edges
	res['max_retweeted_user'] = max_retweeted_user
	res['max_retweeted_number'] = max_retweeted_number
	res['max_retweeter_user'] = max_retweeter_user
	res['max_retweeter_number'] = max_retweeter_number

	writeOutput(json_output_file, res)