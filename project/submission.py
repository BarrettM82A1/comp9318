import numpy as np
from scipy.spatial import distance


def L1_distance(nm, kmp):
    if len(kmp.shape) != 2:
        kmp = kmp[np.newaxis, :]
    return distance.cdist(nm, kmp, 'cityblock')


def kmeans(x, k, max_iter, init):
    for i in range(max_iter):
        labels = np.argmin(L1_distance(x, init), 1)
        for j in range(k):
            if len(x[np.where(labels == j)]) != 0:
                init[j] = np.median(x[np.where(labels == j)], 0)
    labels = np.argmin(L1_distance(x, init), 1)
    return init, labels


def pq(data, P, init_centroids, max_iter):
    list = []
    for i in range(P):
        L = data[:,  i * int(data.shape[1] / P):(i + 1) * int(data.shape[1] / P)]
        list.append(L)
    Data = np.array(list)
    codebooks = []
    codes = []
    for j in range(data.shape[0]):
        codes.append([])
    for i in range(P):
        medians, labels = kmeans(Data[i], init_centroids.shape[1], max_iter, init_centroids[i])
        codebooks.append(medians)
        codes = np.concatenate((codes, labels[:, np.newaxis]), axis=1)
    codebooks = np.array(codebooks)
    codes = np.array(codes)
    return codebooks.astype('float32'), codes.astype('uint8')


def Multi_Sequence(distance, T):
    index = np.argsort(distance)
    point = [index[0]]
    candidate = 1
    while candidate <= T:
        if len(distance) > 0:
            distance[index[0]] = distance[index[candidate]]
        if len(index) > 0:
            point.append(index[candidate])
        candidate += 1
        while distance[index[candidate]] == distance[index[0]]:
            point.append(index[candidate])
            candidate += 1
        if candidate >= T:
            break
    return set(point)


def query(queries, codebooks, codes, T):
    Q = queries.shape[0]
    m = queries.shape[1]
    p = codebooks.shape[0]
    query_list = []
    new_queries = queries.reshape((Q, p, m // p))
    for i in range(Q):
        q = new_queries[i]
        distance = np.zeros((codes.shape[0], 1), dtype=np.float32)
        for j in range(p):
            new_codebooks = codebooks[j]
            new_q = q[j]
            a = L1_distance(new_codebooks, new_q)
            if len(a) > 0:
                distance += a[codes[:, j]]
        if len(distance) > 0:
            new_distance = distance[:, 0]
        query_list.append(Multi_Sequence(new_distance, T))
    return query_list
