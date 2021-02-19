import sys
import collections
import math
import itertools
import bisect
import json

def build_baskets(filename):
    basket_name = None
    basket = []
    baskets = []
    for line in open(input_filename, 'r', encoding='utf-8'):
        tmp = line.split(',')
        user_id, movie_id, rating = tmp[:3]
        if rating != str(5.0):
            continue
        if user_id != basket_name:
            if basket_name is not None:
                baskets.append((basket_name, basket))
            basket = [movie_id]
            basket_name = user_id
        else:
            basket.append(movie_id)
    baskets.append((basket_name, basket))
    return baskets

def get_item_dict(baskets):
    item_dict = {}
    for basket in baskets:
        items = basket[1]
        for item in items:
            if item not in item_dict:
                item_dict[item] = len(item_dict)
    return item_dict

def filter_basket(baskets, item_dict, k):
    if k == 2:
        possible_item = item_dict
    else:
        possible_item = set()
        possible_item = possible_item.union(*item_dict.keys())
    for i in range(len(baskets)):
        basket = baskets[i]
        items = basket[1]
        items_filterd = [item for item in items if item in possible_item]
        baskets[i] = (basket[0], items_filterd)

def inverse_dict(d):
    return {v: k for k, v in d.items()}

def get_possible_k(item_dict, k):
    possible_k = {}
    for pair in itertools.combinations(item_dict.keys(), 2):
        pair_set = set()
        for i in range(2):
            pair_set = pair_set.union(tuple_wrapper(pair[i]))
        if len(pair_set) == k:
            possible_k[frozenset(pair_set)] = [pair[0], pair[1]]
    return possible_k

class FirstList(collections.UserList):
    def __lt__(self, other):
        return self[0].__lt__(other)

def tuple_wrapper(s):
    if type(s) is not tuple:
        s = (s, )
    return s

def tuple_list_method(baskets, support, item_dict, k):
    if item_dict is None:
        item_dict = get_item_dict(baskets)
    else:
        filter_basket(baskets, item_dict, k)
    item_dict_inv = inverse_dict(item_dict)
    n = len(item_dict)
    if k >= 3:
        possible_k = get_possible_k(item_dict, k)        
    tuples = []
    for basket in baskets:
        items = basket[1]
        for kpair in itertools.combinations(items, k):
            if k >= 3:
                pair_set = frozenset(kpair)
                kpair = possible_k.get(pair_set, None)
                if kpair is None:
                    continue
            i = item_dict[kpair[0]]
            j = item_dict[kpair[1]]
            if i > j:
                j, i = i, j   
            idx = i*n+j
            insert_idx = bisect.bisect_left(tuples, idx)
            if insert_idx >= len(tuples):
                tuples.append(FirstList([idx, 1]))
            else:
                tp = tuples[insert_idx] 
                if tp[0] == idx:
                    tp[1] += 1
                else:
                    tuples.insert(insert_idx, FirstList([idx, 1]))
    frequent_itemset_list = []
    for tp in tuples:
        count = tp[1]
        i = tp[0] // n
        j = tp[0] % n
        item_i = item_dict_inv[i]
        item_j = item_dict_inv[j]
        item_all = set()
        for item in (item_i, item_j):
            item_all = item_all.union(tuple_wrapper(item))
        item_all = tuple(sorted(list(item_all)))
        if count >= support:
            frequent_itemset_list.append((item_all, count))
    frequent_itemset_list = sorted(frequent_itemset_list, key=lambda x: [-x[1]] + list(x[0]))
    return frequent_itemset_list

def get_item_counter(baskets):
    item_counter = collections.Counter()
    for basket in baskets:
        items = basket[1]
        item_counter.update(items)
    return item_counter

def get_dict_from_frequent(frequent_list):
    item_dict = {}
    for item in frequent_list:
        item_dict[item] = len(item_dict)
    return item_dict

def apriori(baskets, support, method):
    if type(baskets) is not list:
        baskets = list(baskets)
    item_counter = get_item_counter(baskets)
    itemsets_1 = sorted([(k, v) for k, v in item_counter.items() if v >= support], key=lambda x: x[1], reverse=True)
    frequent_1 = [x[0] for x in itemsets_1]
    itemsets_list = [itemsets_1]
    frequent_list = frequent_1
    frequent_last = frequent_1
    k = 2
    while True:
        item_dict = get_dict_from_frequent(frequent_last)
        itemsets = method(baskets, support, item_dict, k=k)
        if len(itemsets) > 0:
            frequent_last = [x[0] for x in itemsets]
            frequent_list += frequent_last
            itemsets_list.append(itemsets)
            k += 1
        else:
            break
    return itemsets_list

def writeOutput(fp, data):
    with open(fp, 'w') as outfile:
        json.dump(data, outfile)

if __name__ == '__main__':
    # input_filename = './ml-latest-small/ratings_test_truth.csv'
    # output_filename = './testout1.json'
    # interest = 0.2
    # support = 2

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    interest = float(sys.argv[3])
    support = int(sys.argv[4])

    baskets = build_baskets(input_filename)
    num_baskets = len(baskets)
    item_counter = get_item_counter(baskets)
    itemsets_list = apriori(baskets, support, tuple_list_method)
    
    itemsets_counter = {}
    for itemsets in itemsets_list:
        for itemset in itemsets:
            itemsets_counter[itemset[0]] = itemset[1]

    res = []
    for i in range(1, len(itemsets_list)):
        for itemset in itemsets_list[i]:
            cur_support = itemset[1]
            itemset_tuple = itemset[0]
            itemset_set = set(itemset_tuple)
            for item in itemset_set:
                cur_j = item
                itemset_list = list(itemset_tuple)
                itemset_list.remove(item)
                cur_i = itemset_list
                if len(cur_i) > 1:
                    conf = cur_support / itemsets_counter[tuple(cur_i)]
                else:
                    conf = cur_support / itemsets_counter[cur_i[0]]
                pr = item_counter[cur_j] / num_baskets
                if conf - pr > interest:
                    res_cur_i = []
                    for s in cur_i:
                        res_cur_i.append(int(s))
                    res_cur_i.sort()
                    res.append([res_cur_i, int(cur_j), conf - pr, cur_support])
    res.sort(key = lambda x: (-x[2], -x[3], x[0], x[1]))
    # print(len(res))
    writeOutput(output_filename, res)
