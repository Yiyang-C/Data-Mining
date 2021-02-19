import math
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import collections
import itertools

def get_hash_coeffs(br):
    rnds = np.random.choice(2**20, (2, br), replace=False)
    c = 1048583
    return rnds[0], rnds[1], c

def min_hashing(shingles, hash_coeffs, br):
    count = len(shingles)

    (a, b, c) = hash_coeffs
    a = a.reshape(1, -1)
    M = np.zeros((br, count), dtype=int) #Its layout same as slide 56. col are docs, row are signature index
    for i, s in enumerate(shingles):
        # All shingles in the document
        row_idx = np.asarray(list(s)).reshape(-1, 1)
        # Instead of getting many hash functions and run each hash function to each shingles,
        # Use numpy matrix multiplication to apply all hash funcitons to all shingles in the same time
        m = (np.matmul(row_idx, a) + b) % c
        m_min = np.min(m, axis=0) #For each hash function, minimum hash value for all shingles
        M[:, i] = m_min

    return M

def LSH(M, b, r, band_hash_size):
    count = M.shape[1]

    bucket_list = []
    for band_index in range(b):
        # The hash table for each band is stored as a dictionrary of sets. It's more efficient than sparse matrix
        m = collections.defaultdict(set)

        row_start = band_index * r
        for c in range(count):
            v = M[row_start:(row_start+r), c]
            v_hash = hash(tuple(v.tolist())) % band_hash_size
            m[v_hash].add(c)

        bucket_list.append(m)

    return bucket_list

def find_similiar(shingles, query_index, threshold, bucket_list, M, b, r, band_hash_size, verify_by_signature):
    # Step 1: Find candidates
    candidates = set()
    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start+r), query_index]
        v_hash = hash(tuple(v.tolist())) % band_hash_size

        m = bucket_list[band_index]
        bucket = m[v_hash]
#       print(f'Band: {band_index}, candidates: {bucket}')
        candidates = candidates.union(bucket)

#   print(f'Found {len(candidates)} candidates')

    # Step 2: Verify similarity of candidates
    sims = []
    # Since the candidates size is small, we just evaluate it on k-shingles matrix, or signature matrix for greater efficiency
    if verify_by_signature:
        query_vec = M[:, query_index]
        for col_idx in candidates:
            col = M[:, col_idx]
            sim = np.mean(col == query_vec) # Jaccard Similarity is proportional to the fraction of the minhashing signature they agree
            if sim >= threshold:
                sims.append((col_idx, sim))
    else:
        query_set = shingles[query_index]
        for col_idx in candidates:
            col_set = shingles[col_idx]

            sim = len(query_set & col_set) / len(query_set | col_set) # Jaccard Similarity
            if sim >= threshold:
                sims.append((col_idx, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims


if __name__ == '__main__':
    
    movies_filename = sys.argv[1]
    rating_train_filename = sys.argv[2]
    rating_test_filename = sys.argv[3]
    output_filename = sys.argv[4]

    # movies_filename = './ml-latest-small/movies.csv'
    # rating_train_filename = './ml-latest-small/ratings_train.csv'
    # rating_test_filename = 'ml-latest-small/ratings_test.csv'
    # output_filename = './testout.csv'
    # rating_test_truth = pd.read_csv('ml-latest-small/ratings_test_truth.csv')

    movies = pd.read_csv(movies_filename, encoding='utf-8')
    ratings_train = pd.read_csv(rating_train_filename, encoding='utf-8')
    ratings_test = pd.read_csv(rating_test_filename, encoding='utf-8')
    
    threshold = 0.1414
    hash_size = 2**20
    band_hash_size = 2**16
    verify_by_signature = False
    b = 50
    r = 2
    br = b * r

    genres_characters = {'Action': 1,
                     'Adventure': 2,
                     'Animation': 3,
                     'Children': 4,
                     'Comedy': 5,
                     'Crime': 6,
                     'Documentary': 7,
                     'Drama': 8,
                     'Fantasy': 9,
                     'Film-Noir': 10,
                     'Horror': 11,
                     'Musical': 12,
                     'Mystery': 13,
                     'Romance': 14,
                     'Sci-Fi': 15,
                     'Thriller': 16,
                     'War': 17,
                     'Western': 18,
                     'IMAX': 19,
                     '(no genres listed)': 0
                    }

    movie_characters = {}
    movie_index = {}
    movie_index_inverse = {}
    i = 0
    for idx, genres in zip(movies['movieId'], movies['genres']):
        genre_list = genres.split('|')
        # print(idx, genre_list)
        characters_list = []
        for genre in genre_list:
            characters_list.append(genres_characters[genre])
        movie_characters[i] = characters_list
        movie_index[i] = idx
        movie_index_inverse[idx] = i
        i += 1

    shingles = []
    for i in movie_characters:
        shingles.append(set(movie_characters[i]))

    hash_coeffs = get_hash_coeffs(br)
    M = min_hashing(shingles, hash_coeffs, br)
    bucket_list = LSH(M, b, r, band_hash_size)

    movie_index_rating = {}
    for uid, mid, rating in zip(ratings_train['userId'] ,ratings_train['movieId'], ratings_train['rating']):
        movie_index_rating[(uid, movie_index_inverse[mid])] = rating

    result = []
    for index, row in ratings_test.iterrows():
        query_index = movie_index_inverse[row['movieId']]
        query_user = row['userId']
        # print(row['movieId'], query_user)
        sims = find_similiar(shingles, query_index, threshold, bucket_list, M, b, r, band_hash_size, verify_by_signature)
        sims_dict = {}
        for sim in sims:
            mid = movie_index[sim[0]]
            sims_dict[mid] = sim[1]
#         print(sims_dict)
        # print(len(sims_dict))
        mlist = list(ratings_train.loc[ratings_train['userId'] == row['userId']]['movieId'])
        
        filter_mlist = []
#         print(mlist)
        for m in mlist:
            if m in sims_dict:
                filter_mlist.append(m)
        # print(filter_mlist)
#         print(len(filter_mlist))
        ratings_dict = {}
        if not filter_mlist:
            if query_user in ratings_dict:
                result.append(ratings_dict[query_user])
                # print(ratings_dict[query_user])
                # print('=================')
            else:
                mrlist = list(ratings_train.loc[ratings_train['userId'] == row['userId']]['rating'])
                pred_rating = sum(mrlist) / len(mrlist)
                ratings_dict[query_user] = pred_rating
                result.append(pred_rating)
                # print(pred_rating)
                # print('=================')
        else:
#                 filter_ilist = []
#                 for mid in filter_mlist:
#                     filter_ilist.append(movie_index_inverse[mid])
                total = sim_sum = 0
                # target = set(movie_characters[query_index])
                for mid in filter_mlist:
                    sim_sum += sims_dict[mid]
                    idx = movie_index_inverse[mid]
                    # tmp = set(movie_characters[idx])
                    total +=  sims_dict[mid] * movie_index_rating[(query_user, idx)] #0.83
#                     total += (len(target & tmp) / len(target | tmp)) * movie_index_rating[(query_user, idx)] #0.83
#                     total += (len(target & tmp) / len(target | tmp)) * movie_index_rating[(query_user, idx)] * sims_dict[mid]
#                     pred_rating = total / len(filter_ilist)
                pred_rating = total / sim_sum
                result.append(pred_rating)
                # print(pred_rating)
                # print('=================')
    ans = np.array(result)
    # from sklearn.metrics import mean_squared_error
    # mse = mean_squared_error(ans, rating_test_truth['rating'])
    # print(mse)
    # print(ans[:10])
    csv = pd.read_csv(rating_test_filename, encoding='utf-8')
    csv['rating'] = ans
    csv.to_csv(output_filename, index=False)