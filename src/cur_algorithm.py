import numpy as np
import random
import math

from svd_algorithm import SVDAlgorithm

def cur(M, c, r, dim_red = None, repeat=None):
    """
    CUR function returns C,U,R

    Input:
    @M: input numpy array
    @c: Number of column selections
    @r: Number of row selections
    @repeat: Repetition allowed
    """
    column, C = randRows(M.T,100)
    C = C.T
    rows, R = randRows(M,100)
    W = []
    for ri in rows:
        temp = []
        for cj in column:
            temp.append(M[ri][cj])
        W.append(temp)
    W = np.array(W)
    if dim_red == None or dim_red == 1:
        W_u, W, W_v = SVDAlgorithm().svd(W)
    else:
        W_u, W, W_v = SVDAlgorithm().svd(W, dimension_reduction=dim_red)
    for i in range(W.shape[0]):
        W[i][i] = 1 / W[i][i]
    U = np.dot(np.dot(W_v.T, W ** 2), W_u.T)
    M_p = np.dot(np.dot(C,  U), R)
    return M_p


def randRows(M,r):
    total = np.sum((np.square(M)))
    #print(total)
    Row_scores = []
    for column in M:
        temp = np.sum((np.square(column)))
        score = temp/total
        Row_scores.append(score)

    Choices = random.choices(range(M.shape[0]), k=r, weights=Row_scores)
    Choices = sorted(Choices)
    #print(Choices)
    final_rows = []
    for i in Choices:
        temp = M[i]/math.sqrt(r*Row_scores[i])
        final_rows.append(temp)
    final_rows = np.array(final_rows)

    return Choices, final_rows
