# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import logsumexp

# from sklearn.decomposition import LatentDirichletAllocation
from .online_lda import OnlineLatentDirichletAllocation
from sklearn.utils import check_random_state

from _VB_lda import (loglikelihood, _dirichlet_expectation_2d)

EPS = np.finfo(np.float).eps


class LatentDirichletAllocationWithSample(OnlineLatentDirichletAllocation):
    # def __init__(self, *args, **kwargs):
    #     super(LatentDirichletAllocationWithSample, self).__init__(*args, **kwargs)
    #     # self.multinomial_mdl = None
    #     # self.mixture_mdl = None

    def sample_latent_variable(self, X, ifSample=True, ifProbabilistic=True, ifScoredByMLE=True):
        if ifSample and ifProbabilistic:
            rng = check_random_state(self.random_state)

        doc_topic_distr = self.transform(X)
        expect_doc_topic = _dirichlet_expectation_2d(doc_topic_distr)
        doc_topic_distr = (doc_topic_distr.T / np.sum(doc_topic_distr, axis=1)).T
        self.doc_topic_distr = doc_topic_distr

        expect_topic_word_distr =_dirichlet_expectation_2d(self.components_)
        topic_word_distr_d = self.components_ / np.sum(self.components_, axis=1)[:, np.newaxis]

        is_sparse_x = sp.issparse(X)
        n_samples, n_features = X.shape
        n_topics = expect_topic_word_distr.shape[0]


        if is_sparse_x:
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr

        n_k = np.zeros(n_topics)
        n_k_temp = np.zeros(n_topics, dtype=np.int64)
        sample = np.zeros(n_topics, dtype=np.int64)
        n_k_v = np.zeros((n_features, n_topics), dtype=np.int64)
        loglikelihood_Z = 0.0
        loglikelihood_X = 0.0

        if ifScoredByMLE == True:
            for dd in range(n_samples):
                n_k_temp.fill(0)
                if is_sparse_x:
                    ids = X_indices[X_indptr[dd]:X_indptr[dd + 1]]
                    cnts = X_data[X_indptr[dd]:X_indptr[dd + 1]]
                else:
                    ids = np.nonzero(X[dd, :])[0]
                    cnts = X[dd, ids]

                exp_doc_topic_d = expect_doc_topic[dd, :]
                exp_topic_word_d = expect_topic_word_distr[:, ids]
                norm_phi = [logsumexp(exp_doc_topic_d + exp_topic_word_d[:, i]) for i in range(len(ids))]
                # np.dot(exp_doc_topic_d, exp_topic_word_d) + EPS

                for n, num in enumerate(cnts):
                    q_dn = np.exp(exp_doc_topic_d + exp_topic_word_d[:, n] - norm_phi[n])
                    q_dn_total = np.sum(q_dn)
                    if np.sum(1 - q_dn_total) != 1.0:
                        q_dn *= 1.0 / q_dn_total
                    if ifSample:
                        if ifProbabilistic:
                            sample = rng.multinomial(num, q_dn)
                        else:
                            sample.fill(0)
                            sample[np.argmax(q_dn)] = int(num)
                        n_k_temp += sample
                        n_k_v[ids[n]] += sample
                    else:
                        n_k += q_dn * num

                if ifSample:
                    loglikelihood_Z += loglikelihood(n_k_temp)
                    n_k += n_k_temp

            if ifSample:
                for k in range(n_topics):
                    loglikelihood_X += loglikelihood(n_k_v[:, k])
        else:
            for dd in range(n_samples):
                n_k_temp.fill(0)
                if is_sparse_x:
                    ids = X_indices[X_indptr[dd]:X_indptr[dd + 1]]
                    cnts = X_data[X_indptr[dd]:X_indptr[dd + 1]]
                else:
                    ids = np.nonzero(X[dd, :])[0]
                    cnts = X[dd, ids]

                exp_doc_topic_d = expect_doc_topic[dd, :]
                exp_topic_word_d = expect_topic_word_distr[:, ids]
                norm_phi = [logsumexp(exp_doc_topic_d + exp_topic_word_d[:, i]) for i in
                            range(len(ids))]  # np.dot(exp_doc_topic_d, exp_topic_word_d) + EPS

                for n, num in enumerate(cnts):
                    q_dn = np.exp(exp_doc_topic_d + exp_topic_word_d[:, n] - norm_phi[n])
                    if ifSample:
                        if ifProbabilistic:
                            sample = rng.multinomial(num, q_dn)
                        else:
                            sample.fill(0)
                            sample[np.argmax(q_dn)] = int(num)
                        loglikelihood_X -= np.sum(sample * np.log(topic_word_distr_d[:, n] + EPS))
                        n_k_temp += sample
                    else:
                        n_k += q_dn * num
                if ifSample:
                    loglikelihood_Z -= np.sum(n_k_temp * np.log(doc_topic_distr[dd, :] + EPS))
                    n_k += n_k_temp

        if ifSample:
            return n_k, loglikelihood_X, loglikelihood_Z
        else:
            return n_k, None, None

if __name__ == "__main__":
    pass
