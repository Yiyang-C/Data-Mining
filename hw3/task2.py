import sys
import json
import networkx as nx
import itertools
import copy
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readInput(fp):
    data = []
    for line in open(fp, 'r', encoding='utf-8'):
        data.append(json.loads(line))
    return data


if __name__ == '__main__':
    # input_file = './Gamergate.json'
    # input_file = './toy_test/mini_mid_gamergate.json'
    # taskA_output_file = './testoutA.txt'
    # taskB_output_file = './testoutB.txt'
    # taskC_output_file = './testoutC.txt'
    input_file = sys.argv[1]
    taskA_output_file = sys.argv[2]
    taskB_output_file = sys.argv[3]
    taskC_output_file = sys.argv[4]

    tweets = readInput(input_file)
    G = nx.Graph()

    for tweet in tweets:
        if 'retweeted_status' not in tweet:
            G.add_node(tweet['user']['screen_name'])
        else:
            if G.has_edge(tweet['user']['screen_name'], tweet['retweeted_status']['user']['screen_name']):
                G[tweet['user']['screen_name']][tweet['retweeted_status']['user']['screen_name']]['weight'] += 1
            else:
                G.add_edge(tweet['user']['screen_name'], tweet['retweeted_status']['user']['screen_name'], weight = 1)

    edge_betweenness = nx.edge_betweenness_centrality(G,normalized=False)
    edge_betweenness_sort = sorted(edge_betweenness.items(), key=lambda x:x[1], reverse=True)

    # i = 0
    original_G = copy.deepcopy(G)
    max_modularity = -float('inf')
    m = original_G.size(weight='weight')
    while G.number_of_edges() > 0:
        # print(i, G.number_of_edges(), )
        # i += 1
        edge_betweenness_list = nx.edge_betweenness_centrality(G,normalized=False, weight='weight')
        edge_betweenness_sorted_list = sorted(edge_betweenness_list.items(), key=lambda x:x[1], reverse=True)
        max_edge_betweenness = edge_betweenness_sorted_list[0][1]
        for edge_betweenness in edge_betweenness_sorted_list:
            if edge_betweenness[1] == max_edge_betweenness:
                G.remove_edge(edge_betweenness[0][0], edge_betweenness[0][1])
            else:
                break
        tmp_modularity = 0
        for partition in nx.connected_components(G):
            if len(partition) == 1:
                continue
            else:
                for node1 in partition:
                    for node2 in partition:
                        if node1 == node2:
                            continue
                        if not G.has_edge(node1, node2):
                            tmp_modularity += (-original_G.degree(weight='weight')[node1] * original_G.degree(weight='weight')[node2] / (m * 2))
                        else:
                            tmp_modularity += (G[node1][node2]['weight'] - original_G.degree(weight='weight')[node1] * original_G.degree(weight='weight')[node2] / (m * 2))
        cur_modularity = tmp_modularity / (m * 2)
        # print(len([c for c in nx.connected_components(G)]), cur_modularity)
        if cur_modularity > max_modularity:
            max_modularity = cur_modularity
            # print('find larger modularity !!!!!!!!!!!!!!!!!')
            optimal_G = copy.deepcopy(G)

    # i = 1
    community_list = []
    for partition in nx.connected_components(optimal_G):
        community = list(partition)
        community.sort()
        community_list.append(community)
        # print(i)
        # print('==========')
        # print(partition)
        # print('==========')
        # i+=1
    community_list.sort()
    community_list.sort(key=lambda x: len(x))
    txtA = community_list

    file = open(taskA_output_file, 'w', encoding='utf-8')
    file.write('Best Modularity is: ' + str(max_modularity) + '\n')
    for user_list in txtA:
        i = 0
        for user in user_list:
            if i == 0:
                file.write('\'' +user + '\'')
            else:
                file.write(',\'' +user + '\'')
            i += 1
        file.write('\n')
    file.close()

    community_A, community_B = community_list[-1], community_list[-2]
    community_A_set = set(community_A)
    community_B_set = set(community_B)

    user_tweets = collections.defaultdict(str)
    for tweet in tweets:
        user_tweets[tweet['user']['screen_name']] += ( ' ' + tweet['text'])
        if 'retweeted_status' in tweet:
            user_tweets[tweet['retweeted_status']['user']['screen_name']] += ( ' ' + tweet['retweeted_status']['text'])

    train_data = []
    train_label = []
    for user in community_A:
        train_data.append(user_tweets[user])
        train_label.append(1)
    for user in community_B:
        train_data.append(user_tweets[user])
        train_label.append(2)

    vectorizer = TfidfVectorizer()
    train_data_tfidf = vectorizer.fit_transform(train_data)
    train_label_nparray = np.asarray(train_label)

    clf = MultinomialNB().fit(train_data_tfidf, train_label_nparray)

    test_data = []
    test_data_user = []
    for user in user_tweets.keys():
        if (user not in community_A_set) and (user not in community_B_set):
            test_data.append(user_tweets[user])
            test_data_user.append(user)

    test_data_tfidf = vectorizer.transform(test_data)
    predicted = clf.predict(test_data_tfidf)

    community_A_res = [] + community_A
    community_B_res = [] + community_B
    # cnt_A = cnt_B = 0
    for user, predict_res in zip(test_data_user, predicted):
        if predict_res == 1:
            community_A_res.append(user)
            # cnt_A += 1
        else:
            community_B_res.append(user)
            # cnt_B += 1

    community_A_res.sort()
    community_B_res.sort()

    file = open(taskB_output_file, 'w', encoding='utf-8')
    i = 0
    for user in community_A_res:
        if i == 0:
            file.write('\'' +user + '\'')
        else:
            file.write(',\'' +user + '\'')
        i += 1
    file.write('\n')
    i = 0
    for user in community_B_res:
        if i == 0:
            file.write('\'' +user + '\'')
        else:
            file.write(',\'' +user + '\'')
        i += 1
    file.close()

    count_vect = CountVectorizer()
    train_data_CV = count_vect.fit_transform(train_data)

    clf_CV = MultinomialNB().fit(train_data_CV, train_label_nparray)

    test_data_CV = count_vect.transform(test_data)
    predicted_CV = clf_CV.predict(test_data_CV)

    community_A_res_CV = [] + community_A
    community_B_res_CV = [] + community_B
    for user, predict_res in zip(test_data_user, predicted_CV):
        if predict_res == 1:
            community_A_res_CV.append(user)
        else:
            community_B_res_CV.append(user)

    community_A_res_CV.sort()
    community_B_res_CV.sort()

    file = open(taskC_output_file, 'w', encoding='utf-8')
    i = 0
    for user in community_A_res_CV:
        if i == 0:
            file.write('\'' +user + '\'')
        else:
            file.write(',\'' +user + '\'')
        i += 1
    file.write('\n')
    i = 0
    for user in community_B_res_CV:
        if i == 0:
            file.write('\'' +user + '\'')
        else:
            file.write(',\'' +user + '\'')
        i += 1
    file.close()
    
