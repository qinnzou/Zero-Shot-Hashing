import numpy as np

"""
query_h: hash codes of query set [1/-1]
database_h: hash codes of database set [1/-1]
Wt: indice matrix of similarity between query and database set
top_nums: the map of top_k retrieved samples should be returned
"""


def evaluation(query_h, database_h, Wt, St, top_nums):
    query_num = query_h.shape[0]
    database_num = database_h.shape[0]
    nbits = query_h.shape[1]
    D2 = np.dot(query_h, np.transpose(database_h))
    ham_dist = (nbits - D2)/2.

    sort_indices = np.argsort(ham_dist, axis=1)
    maximum_indices = np.argsort(-St, axis=1)
    query_sorted_Wt = np.zeros(shape=Wt.shape, dtype=np.float32)
    query_sorted_St = np.zeros(shape=St.shape, dtype=np.float32)
    maximum_sorted_St = np.zeros(shape=St.shape, dtype=np.float32)

    for i in range(query_num):
        query_sorted_Wt[i, :] = Wt[i, sort_indices[i, :]]
        maximum_sorted_St[i, :] = St[i, maximum_indices[i, :]]
        query_sorted_St[i, :] = St[i, sort_indices[i, :]]


    # ap = np.zeros(shape=(query_num, len(top_nums)), dtype=np.float32)
    # ap2 = np.zeros(shape=(query_num, len(top_nums)), dtype=np.float32)
    map = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    ndcg = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    acg = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    presicion = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    recall = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    wap = np.zeros(shape=(len(top_nums)), dtype=np.float32)

    cum_Wt = np.cumsum(query_sorted_Wt, axis=1)
    cum_St = np.cumsum(query_sorted_St, axis=1)

    c = np.tile(np.reshape(range(1, database_num+1), (1, database_num)), (query_num, 1))
    cum_Wt_div = np.true_divide(cum_Wt, c)
    cum_St_div = np.true_divide(cum_St, c)

    dcg = np.true_divide((2**query_sorted_St)-1, np.log(c+1))
    maximum_dcg = np.true_divide((2**maximum_sorted_St)-1, np.log(c+1))

    for ii in range(len(top_nums)):
        topk = top_nums[ii]
        tmp_cum_Wt_div = cum_Wt_div[:, 0:topk]
        tmp_cum_St_div = cum_St_div[:, 0:topk]
        tmp_sorted_Wt = query_sorted_Wt[:, 0:topk]
        tmp_dcg = dcg[:, 0:topk]
        tmp_maximum_dcg = maximum_dcg[:, 0:topk]

        # for i in range(query_num):
        #     if np.sum(tmp_sorted_Wt[i, :]) > 0:
        #         ap[i, ii] = np.sum(np.multiply(tmp_cum_Wt_div[i, :], tmp_sorted_Wt[i, :])) / np.sum(tmp_sorted_Wt[i, :])
        #         ap2[i, ii] = np.sum(np.multiply(tmp_cum_St_div[i, :], tmp_sorted_Wt[i, :])) / np.sum(tmp_sorted_Wt[i, :])
        #     else:
        #         ap[i, ii] = 0
        #         ap2[i, ii] = 0
        map[ii] = np.mean(np.nan_to_num(np.true_divide(np.sum(np.multiply(tmp_cum_Wt_div, tmp_sorted_Wt), axis=1), np.sum(tmp_sorted_Wt, axis=1))))
        wap[ii] = np.mean(np.nan_to_num(np.true_divide(np.sum(np.multiply(tmp_cum_St_div, tmp_sorted_Wt), axis=1), np.sum(tmp_sorted_Wt, axis=1))))

        _dcg = np.true_divide(np.sum(tmp_dcg, axis=1), np.sum(tmp_maximum_dcg, axis=1))
        # _dcg[np.isnan(_dcg)] = 0
        ndcg[ii] = np.mean(_dcg)
        acg[ii] = np.mean(query_sorted_St[:, 0:topk])
        # map[ii] = np.mean(ap[:, ii])
        # wap[ii] = np.mean(ap2[:, ii])
        presicion[ii] = np.mean(tmp_sorted_Wt)
        recall[ii] = np.true_divide(np.sum(tmp_sorted_Wt), np.sum(query_sorted_Wt))
    return presicion, recall, map, wap, acg, ndcg

