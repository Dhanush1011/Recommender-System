import os
import time
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from svd_algorithm import SVDAlgorithm
from measures import *
from cur_algorithm import *
from collaborate import *

def load_dataset():
    df=pd.read_csv('./dataset/ratings.csv', names=['user_id', 'movie_id', 'rating'], sep='\t', header=None, engine = 'python', usecols=range(3))
    train, test = train_test_split(df, test_size=0.2)
    train_mat = np.ndarray(shape=(np.max(df.movie_id.values), np.max(df.user_id.values)))
    test_mat = np.ndarray(shape=(np.max(df.movie_id.values), np.max(df.user_id.values)))
    train_mat[train.movie_id.values-1, train.user_id.values-1] = train.rating.values
    test_mat[test.movie_id.values-1, test.user_id.values-1] = test.rating.values
    return train_mat, test_mat


def run_collaborative_filtering(M, test):
    print("\n******************* Collaborative Filtering *******************")
    start = time.time()
    M = M.T
    test = test.T
    cf = Collaborate(M.T)
    M_p = cf.fill()
    print("Time: " +str(time.time() - start))
    print("RMSE: " + str(rmse(test, M_p.T)))
    print("Top K precision: " + str(top_k(40, test, M_p.T)))
    print("Spearman correlation: " + str(spearman_correlation(test, M_p.T)))

def run_collaborative_filtering_baseline(M, test):
    print("\n******************* Collaborative Filtering with baseline *******************")
    start = time.time()
    M = M.T
    test = test.T
    cfb = Collaborate(M.T)
    M_p = cfb.fill(baseline=True)
    print("Time: " +str(time.time() - start))
    print("RMSE: " + str(rmse(test, M_p.T)))
    print("Top K precision: " + str(top_k(40, test, M_p.T)))
    print("Spearman correlation: " + str(spearman_correlation(test, M_p.T)))

def run_svd(M, test):
    print("\n******************* SVD *******************")
    s = SVDAlgorithm()
    svd_start = time.time()
    U, sigma, V = s.svd(M, dimension_reduction=1.0)
    M_p = np.dot(np.dot(U, sigma), V)
    print("Time: " +str(time.time() - svd_start))
    print("RMSE: " + str(rmse(test, M_p)))
    print("Top K precision: " + str(top_k(40, test, M_p)))
    print("Spearman correlation: " + str(spearman_correlation(test, M_p)))

def run_svd_reduce(M, test):
    print("\n******************* SVD Reduction *******************")
    s = SVDAlgorithm()
    svd_reduce_start = time.time()
    U, sigma, V = s.svd(M, dimension_reduction=0.9)
    M_p = np.dot(np.dot(U, sigma), V)
    print("Time: " +str(time.time() - svd_reduce_start))
    print("RMSE: " + str(rmse(test, M_p)))
    print("Top K precision: " + str(top_k(40, test, M_p)))
    print("Spearman correlation: " + str(spearman_correlation(test, M_p)))

def run_cur(M, test):
    print("\n******************* CUR *******************")
    cur_start = time.time()
    M_p = cur(M, 600, 600, repeat=False)
    print("Time: " +str(time.time() - cur_start))
    print("RMSE: " + str(rmse(test, M_p)/100))
    print("Top K precision: " + str(top_k(40, test, M_p)))
    print("Spearman correlation: " + str(spearman_correlation(test, M_p)))

def run_cur_reduce(M, test):
    print("\n******************* CUR Reduction *******************")
    cur_reduce_start = time.time()
    M_p = cur(M, 600, 600, dim_red=0.9, repeat=False)
    print("Time: " +str(time.time() - cur_reduce_start))
    print("RMSE: " + str(rmse(test, M_p)))
    print("Top K precision: " + str(top_k(40, test, M_p)))
    print("Spearman correlation: " + str(spearman_correlation(test, M_p)))


if __name__=="__main__":
    train, test = load_dataset()
    print(train, "\n")
    mean = train[train!=0].mean(axis=None)
    train[train==0] = mean
    print(train)
    # run_collaborative_filtering(train,test)
    # run_collaborative_filtering_baseline(train,test)
    run_svd(train,test)
    # run_svd_reduce(train,test)
    # run_cur(train,test)
    # run_cur_reduce(train,test)
