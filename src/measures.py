import numpy as np


def rmse(M, M_p):
    """
    Computes Root Mean Square Error.

    Input:
    @M - Actual numpy array
    @M_p - Predicted numpy array

    Returns: Root Mean square error - float
    """
    x_len = M.shape[0]
    y_len = M.shape[1]
    error = 0
    N = x_len*y_len
    for i in range(x_len):
        for j in range(y_len):
            error += ((M[i][j] - M_p[i][j]) ** 2)/N
    error = (error) ** 0.5
    return error



def top_k(k, M, M_p):
    precision_list = []
    threshold = 2
    for i in range(M.shape[0]):
        rating_dict = {}
        for j in range(M.shape[1]):
            rating_dict[j] = [M_p[i][j], M[i][j]]
        # print(rating_dict)
        var = {k: v for k, v in sorted(rating_dict.items(), key=lambda item: item[1], reverse=True)}
        count = 0
        rel_recom = 0
        for i in var.keys():
            if count<k:
                count += 1
                if var[i][1] > threshold:
                    rel_recom += 1

        temp = rel_recom/k
        #print(temp)
        precision_list.append(temp)

    avg_precision = np.average(precision_list)

    return avg_precision

def spearman_correlation(M, M_p):
    """
    Returns Spearman score for the prediction.
    Formula: 1 - [sum(diff(predicted - actual)^2) / n((n^2)-1)]

    Input:
    @M - Actual numpy array.
    @M_p - Predicted numpy array.

    Returns:
    Spearman score - float
    """
    x_len = M.shape[0]
    y_len = M.shape[1]
    N = 0
    sum = 0
    for i in range(x_len):
        for j in range(y_len):
            if M[i][j] != 0:
                N += 1
                sum += (M[i][j] - M_p[i][j]) ** 2
    N = (N*(N**2 - 1))
    sum = 1 - (sum/N)
    return sum
