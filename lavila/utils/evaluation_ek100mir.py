# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of the code is from
# `https://github.com/mwray/Joint-Part-of-Speech-Embeddings/tree/main/src/evaluation/NDCG.py`
# and
# `https://github.com/mwray/Joint-Part-of-Speech-Embeddings/tree/main/src/evaluation/mAP.py`
# Modified by Yue Zhao

import numpy as np


def calculate_DCG(similarity_matrix, relevancy_matrix, k_counts):
    """
    Calculates the Discounted Cumulative Gain (DCG) between two modalities for
    the first modality.
    DCG = \sum_{i=1}^k \frac{rel_i}{log_2(i + 1)}
    i.e. the sum of the k relevant retrievals which is calculated as the scaled
    relevancy for the ith item. The scale is designed such that early
    retrievals are more important than later retrievals.
    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith,jth] element is the predicted similarity
          between the ith item from the first modality and the jth item from
          the second modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    Returns:
        - The DCG for each item in the first modality, a n1 length vector.
    """
    x_sz, y_sz = similarity_matrix.shape
    ranks = np.argsort(similarity_matrix)[:, ::-1]
    # Create vector of size (n,) where n is the length of the last dimension in
    # similarity matrix
    # This vector is of the form log(i+1)
    logs = np.log2(np.arange(y_sz) + 2)
    # Convert logs into the divisor for the DCG calculation, of size similarity
    # matrix
    divisors = np.repeat(np.expand_dims(logs, axis=0), x_sz, axis=0)

    # mask out the sorted relevancy matrix to only use the first k relevant
    # retrievals for each item.
    columns = np.repeat(np.expand_dims(np.arange(x_sz), axis=1), y_sz, axis=1)
    numerators = relevancy_matrix[columns, ranks] * k_counts
    # Calculate the final DCG score (note that this isn't expected to sum to 1)
    return np.sum(numerators / divisors, axis=1)


def calculate_k_counts(relevancy_matrix):
    """
    Works out the maximum number of allowed retrievals when working out the
    Discounted Cumulative Gain. For each query the DCG only uses the first k
    items retrieved which constitute the k relevant items for that query
    (otherwise the nDCG scores can be deceptively high for bad rankings).
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality.  The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
    Returns:
        - Matrix of size n1 x n2 (see relevancy matrix for more info). This is
          created as a mask such that if the [ith, jth] element is 1 it
          represents a valid item to use for the calculation of DCG for the
          ith item after sorting. For example, if relevancy matrix of:
        [[1, 0.5, 0],
          [0, 0  , 1]]
          is given, then the k_counts matrix will be:
        [[1, 1, 0],
         [1, 0, 0]]
         i.e. the first row has 2 non-zero items, so the first two retrieved
         items should be used in the calculation. In the second row there is
         only 1 relevant item, therefore only the first retrieved item should
         be used for the DCG calculation.
    """
    return (np.sort(relevancy_matrix)[:, ::-1] > 0).astype(int)


def calculate_IDCG(relevancy_matrix, k_counts):
    """
    Calculates the Ideal Discounted Cumulative Gain (IDCG) which is the value
    of the Discounted Cumulative Gain (DCG) for a perfect retrieval, i.e. the
    items in the second modality were retrieved in order of their descending
    relevancy.
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    """
    return calculate_DCG(relevancy_matrix, relevancy_matrix, k_counts)


def calculate_nDCG(similarity_matrix, relevancy_matrix, k_counts=None, IDCG=None, reduction='mean'):
    """
    Calculates the normalised Discounted Cumulative Gain (nDCG) between two
    modalities for the first modality using the Discounted Cumulative Gain
    (DCG) and the Ideal Discounted Cumulative Gain (IDCG).
    nDCG = \frac{DCG}{IDCG}
    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the second
          modality. The [ith,jth] element is the predicted similarity between
          the ith item from the first modality and the jth item from the second
          modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: optional parameter: matrix of size n1 x n2 (see
          similarity_matrix above) which includes information on which items to
          use to calculate the DCG for (see calculate_k_counts for more info on
          this matrix). This will be calculated using calculate_IDCG if not
          present, but should be pre-processed for efficiency.
        - IDCG: Optional parameter which includes the pre-processed Ideal
          Discounted Cumulative Gain (IDCG). This is a vector of size n1 (see
          similarity_matrix above) which contains the IDCG value for each item
          from the first modality. This will be calculated using calculate_IDCG
          if not present, but should be pre-processed for efficiency.
        - reduction: what to use to reduce the different nDCG scores. By
          default this applies np.mean across all different queries.
    Returns:
        - The nDCG values for the first modality.
    """
    if k_counts is None:
        k_counts = calculate_k_counts(relevancy_matrix)
    DCG = calculate_DCG(similarity_matrix, relevancy_matrix, k_counts)
    if IDCG is None:
        IDCG = calculate_IDCG(relevancy_matrix, k_counts)
    if reduction == 'mean':
        return np.mean(DCG / IDCG)
    elif reduction is None:
        return DCG / IDCG


def calculate_mAP(sim_mat, relevancy_matrix):
    """
    Computes the mean average precision according to the following formula of
    average precision:
    \frac{\sum_{k=1}^n p(k) x rel(k)}{num_rel_docs}
    where p(k) is the precision at k, rel(k) is an indicator function
    determining whether the kth returned item is relevant or not and
    num_rel_docs is the number of relevant items to find within the search.
    The mean average precision is the mean of the average precision for each
    query item (i.e row in the matrix)
    This function takes in two parameters:
        - sim_mat: a NxM matrix which represents the similarity between two
        modalities (with modality 1 being of size N and modality 2 of size M).
        - relevancy_matrix: an NxM matrix which represents the relevancy between two
        modalities of items (with modality 1 being of size N and modality 2 of
        size M).
    """
    # Find the order of the items in modality 2 according to modality 1
    ranked_order = (-sim_mat).argsort()
    ranked_sim_mat = sim_mat[np.arange(sim_mat.shape[0])[:, None], ranked_order]
    # re-order the relevancy matrix to accommodate the proposals
    ranked_rel_mat = relevancy_matrix[np.arange(relevancy_matrix.shape[0])[:, None], ranked_order]

    # find the number of relevant items found at each k
    cumulative_rel_mat = np.cumsum(ranked_rel_mat, axis=1)
    # Mask this ensuring that it is non zero if the kth term is 1 (rel(k) above)
    cumulative_rel_mat[ranked_rel_mat != 1] = 0
    # find the divisor for p(k)
    divisor = np.arange(ranked_rel_mat.shape[1]) + 1

    # find the number of relevant docs per query item
    number_rel_docs = np.sum(ranked_rel_mat == 1, axis=1)

    # find the average precision per query, within np.sum finds p(k) * rel(k)
    avg_precision = np.sum(cumulative_rel_mat / divisor, axis=1) / number_rel_docs
    mAP = np.mean(avg_precision)
    return mAP


def get_mAP(similarity_matrix, rel_matrix):
    vis_map = calculate_mAP(similarity_matrix, rel_matrix)
    txt_map = calculate_mAP(similarity_matrix.T, rel_matrix.T)
    return vis_map, txt_map, (vis_map + txt_map) / 2


def get_nDCG(similarity_matrix, rel_matrix):
    vis_k_counts = calculate_k_counts(rel_matrix)
    txt_k_counts = calculate_k_counts(rel_matrix.T)
    vis_IDCG = calculate_IDCG(rel_matrix, vis_k_counts)
    txt_IDCG = calculate_IDCG(rel_matrix.T, txt_k_counts)
    vis_nDCG = calculate_nDCG(similarity_matrix, rel_matrix, k_counts=vis_k_counts, IDCG=vis_IDCG)
    txt_nDCG = calculate_nDCG(similarity_matrix.T, rel_matrix.T, k_counts=txt_k_counts, IDCG=txt_IDCG)
    return vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2
