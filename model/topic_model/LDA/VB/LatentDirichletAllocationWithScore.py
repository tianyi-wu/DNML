# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gammaln
from scipy.stats import dirichlet

from .LatentDirichletAllocationWithSample import LatentDirichletAllocationWithSample


class LatentDirichletAllocationWithScore(LatentDirichletAllocationWithSample):
    def score_new(self, X, y=None, n_sample_trial=5):
        complexities = self._sample_and_calculate_score(X, n_sample_trial=n_sample_trial)
        return complexities

    def _sample_and_calculate_score(self, X, n_sample_trial=5):
        K, V = self.exp_dirichlet_component_.shape
        D = X.shape[0]
        n_d = [int(nd + 1) for nd in np.sum(X, axis=1)]

        new_2part_complexity_array = np.zeros(n_sample_trial)
        new_2part_complexity_Z = np.sum([self.multinomial_mdl.calculate_mdl(nd, K) for nd in n_d if nd > 0])
        completed_loglikelihood_X_array = np.zeros(n_sample_trial)
        completed_loglikelihood_Z_array = np.zeros(n_sample_trial)
        for idx in range(n_sample_trial):
            n_k, loglikelihood_X, loglikelihood_Z = self.sample_latent_variable(X, ifSample=True, ifProbabilistic=True,
                                                                                ifScoredByMLE=True)
            completed_loglikelihood_X_array[idx] = loglikelihood_X
            completed_loglikelihood_Z_array[idx] = loglikelihood_Z

            new_2part_complexity_array[idx] = new_2part_complexity_Z + np.sum(
                [self.multinomial_mdl.calculate_mdl(int(nk + 1), V) for nk in n_k if int(nk + 1) > 0])
            assert abs(np.sum(X) - np.sum(n_k)) < 5, print(np.sum(X), np.sum(n_k), n_k)

        completed_loglikelihood_array = completed_loglikelihood_X_array + completed_loglikelihood_Z_array
        new_2part_complexity = np.mean(new_2part_complexity_array)
        old_mdl_rissanen_complexity = self.mixture_mdl.calculate_mdl(np.mean(n_d), self.n_topics,
                                                                     n_free_para=self._n_parameters())

        # bic_num = np.log(np.sum(_X)) / 2
        bic_num = np.log(np.mean(np.sum(X, axis=1))) / 2

        doc_topic_distr = self.transform(X)
        variational_bound = self._approx_bound(X, doc_topic_distr, sub_sampling=False)

        loglikelihood = self.score(X)
        return {
            'old_rissanen_complete': (completed_loglikelihood_array + old_mdl_rissanen_complexity).tolist(),
            'old_rissanen_penalty': [old_mdl_rissanen_complexity],

            'new_em_complete': (completed_loglikelihood_array + new_2part_complexity_array).tolist(),
            'new_em': [-loglikelihood + new_2part_complexity],

            'aic_complete': (completed_loglikelihood_array + self._n_parameters()).tolist(),
            'aic_penalty': [self._n_parameters()],

            'bic_complete': (completed_loglikelihood_array + self._n_parameters() * bic_num).tolist(),
            'bic_penalty': [self._n_parameters() * bic_num],

            'variational_bound': [-variational_bound],
        }
