import math
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import collections
import itertools

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

    movie_id = movies['movieId'].unique()
    movie_len = len(movie_id)

    user_id = ratings_train['userId']
    user_id.append(ratings_test['userId'], ignore_index=True)
    user_id = user_id.unique()
    user_len = len(user_id)

    movie_index = {}
    movie_index_inverse = {}
    i = 0
    for idx in movies['movieId']:
        movie_index[i] = idx
        movie_index_inverse[idx] = i
        i += 1

    X = np.zeros((movie_len, user_len))

    total_mean = sum(ratings_train['rating']) / len(ratings_train['rating'])

    user_rating_mean = {}
    for uid in range(user_len):
        mrlist = list(ratings_train.loc[ratings_train['userId'] == uid + 1]['rating'])
        if mrlist:
            mean = sum(mrlist) / len(mrlist)
            user_rating_mean[uid] = mean

    movie_rating_mean = {}
    for mid in range(movie_len):
        mrlist = list(ratings_train.loc[ratings_train['movieId'] == movie_index[mid]]['rating'])
        # print(movie_index[mid], len(mrlist))
        if len(mrlist) > 1:
            mean = sum(mrlist) / len(mrlist)
            movie_rating_mean[mid] = mean

    X = np.zeros((movie_len, user_len))

    movie_user_rating = {}
    for uid, mid, rating in zip(ratings_train['userId'] ,ratings_train['movieId'], ratings_train['rating']):
        i = movie_index_inverse[mid]
        j = uid - 1
        movie_user_rating[(i, j)] = rating
        X[i][j] = rating

    for i in range(movie_len):
        for j in range(user_len):
            if X[i][j] == 0:
                if (i in movie_rating_mean) and (j in user_rating_mean):
                    X[i][j] = (movie_rating_mean[i] + user_rating_mean[j]) / 2
                elif (i not in movie_rating_mean) and (j in user_rating_mean):
                    X[i][j] = user_rating_mean[j]
                elif (i in movie_rating_mean) and (j not in user_rating_mean):
                    X[i][j] = movie_rating_mean[i]
                else:
                    X[i][j] = total_mean

    u, s, vh = np.linalg.svd(X)
    s = np.sqrt(s)
    s_u = np.zeros(X.shape)
    for i, v in enumerate(s):
        s_u[i, i] = v

    s_v = np.diag(s)

    r = 10
    A = np.matmul(u, s_u[:, :r])
    B = np.matmul(s_v[:r, :], vh)
    AB = np.matmul(A, B)
    loss = np.mean((AB - X)**2)

    X = AB
    M_avg_movie_rating = np.average(X, axis = 1)
    M_avg_rating = np.average(X, axis = 0)

    new_movie_rating_mean = {}
    new_user_rating_mean = {}
    for i in range(movie_len):
        mean = sum(X[i]) / user_len
        new_movie_rating_mean[i] = mean
    for j in range(user_len):
        total = 0
        for i in range(movie_len):
            total += X[i][j]
        mean = total / movie_len
        new_user_rating_mean[j] = mean
    new_total_mean = np.average(X)

    res = []
    for uid, mid in zip(ratings_test['userId'], ratings_test['movieId']):
        i = movie_index_inverse[mid]
        j = uid - 1
        if (i in movie_rating_mean) and (j in user_rating_mean):
                    X[i][j] += movie_rating_mean[i] + user_rating_mean[j] - new_movie_rating_mean[i] - new_user_rating_mean[j]
        elif (i not in movie_rating_mean) and (j in user_rating_mean):
            X[i][j] += user_rating_mean[j] - new_user_rating_mean[j]
        elif (i in movie_rating_mean) and (j not in user_rating_mean):
            X[i][j] += movie_rating_mean[i] - new_movie_rating_mean[i]
        else:
            X[i][j] += total_mean - new_total_mean
            
        if X[i][j] < 0.5:
            res.append(0.5)
        elif X[i][j] > 5:
            res.append(5)
        else:
            res.append(X[i][j])

    ans = np.array(res)
    
    # from sklearn.metrics import mean_squared_error
    # mse = mean_squared_error(ans, rating_test_truth['rating'])
    # print(mse)

    csv = pd.read_csv(rating_test_filename, encoding='utf-8')
    csv['rating'] = ans
    csv.to_csv(output_filename, index=False)