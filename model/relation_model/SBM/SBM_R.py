from collections import Counter

import numpy as np
import readline
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from scipy.special import gammaln
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.cluster import normalized_mutual_info_score

from ...topic_model.LDA.VB.LatentDirichletAllocationWithSample import loglikelihood as multinomial_mle_log_likelihood

readline

blockmodels = importr("blockmodels")
base = importr("base")
dollar = base.__dict__["$"]

POSITIVE_VALUE = 1
NEGATICE_VALUE = 0
MISSING_VALUE = -1

EPS = np.finfo(np.float).eps


def convert_to_index(z, n):
    return z[:, n].astype(bool) if len(z.shape) == 2 else (z == n)


def sample_category(z):
    a = range(z.shape[1])
    return np.array([np.random.choice(a, 1, p=p)[0] for p in z])


def check_latent_index_variable(z):
    unique_z = sorted(np.unique(z))
    if len(unique_z) == np.max(z) + 1:
        return z
    new_z = np.zeros(z.shape, dtype=np.int64)
    for index, current in enumerate(unique_z):
        new_z[z == current] = index
    return new_z


def convert_samples_to_counts(arr, size):
    counts = np.zeros(size, dtype=np.int64)
    for index, count in Counter(arr).items():
        counts[index] = count
    return counts


def log_likelihood_low_bound(X, pi, eta, z, index_pos_for_z1, index_neg_for_z1):
    return np.sum(z * np.log(pi)[np.newaxis, :]) + \
           np.sum((z @ np.log(eta) @ z.T) * index_pos_for_z1) + \
           np.sum((z @ np.log(1.0 - eta) @ z.T) * index_neg_for_z1)


def _log_likelihood_mle(X, z, if_ss=False):
    z = check_latent_index_variable(z)
    n_clusters = np.max(z) + 1
    n_z = convert_samples_to_counts(z, n_clusters)

    log_z = 0.0
    log_z += - multinomial_mle_log_likelihood(n_z)

    log_x = 0.0
    for k in range(n_clusters):
        for l in range(n_clusters):
            one_num = np.sum(X[z == k, :][:, z == l] == POSITIVE_VALUE)
            mone_num = np.sum(X[z == k, :][:, z == l] == NEGATICE_VALUE)
            log_x += - multinomial_mle_log_likelihood(np.array((one_num, mone_num), dtype=np.int64))
    if if_ss:
        return log_z + log_x, n_z
    else:
        return log_z + log_x


def _log_likelihood_fitted(X, z, pi, eta, if_ss=False):
    n_clusters = pi.shape[0]
    n_z = convert_samples_to_counts(z, n_clusters)

    log_z = 0.0
    log_z += n_z * np.log(pi)

    log_x = 0.0
    for k in range(n_clusters):
        for l in range(n_clusters):
            if n_z[k] > 0 and n_z[l] > 0:
                total_num = np.sum(z == k) * np.sum(z == l)
                one_num = X[z == k, :][:, z == l]
                log_x += one_num * np.log(eta[k][l]) + (total_num - one_num) * np.log(1.0 - eta[k][l])
    if if_ss:
        return log_z + log_x, n_z
    else:
        return log_z + log_x


def calculate_purity(X, z, n_clusters):
    total_num = 0
    purity_num = 0
    for n in range(n_clusters):
        for m in range(n_clusters):
            p1_num = np.sum(X[convert_to_index(z, n), :][:, convert_to_index(z, m)] == POSITIVE_VALUE)
            m1_num = np.sum(X[convert_to_index(z, n), :][:, convert_to_index(z, m)] == NEGATICE_VALUE)
            total_num += p1_num + m1_num
            purity_num += max(p1_num, m1_num)
    return - purity_num / total_num


def calculate_entropy(X, z, n_clusters):
    total_num = 0
    entropy = 0.0
    for n in range(n_clusters):
        for m in range(n_clusters):
            p1_num = np.sum(X[convert_to_index(z, n), :][:, convert_to_index(z, m)] == POSITIVE_VALUE)
            m1_num = np.sum(X[convert_to_index(z, n), :][:, convert_to_index(z, m)] == NEGATICE_VALUE)
            cluster_num = p1_num + m1_num
            total_num += cluster_num
            for num in [p1_num, m1_num]:
                if num > 0:
                    entropy -= num * np.log(num / cluster_num)
    return entropy / total_num


class TrainedSBM(BaseEstimator, TransformerMixin):
    def __init__(self, pi, eta, z_posterior):
        self.n_clusters = eta.shape[0]
        self.pi = pi
        self.eta = eta
        self.z_posterior = z_posterior

    def _n_parameters(self):
        return self.n_clusters - 1 + self.n_clusters * self.n_clusters

    def score_new(self, X, calculators, y=None, true_z=None, n_sample_trial=5):
        multinomial_mdl = calculators["multinomial_mdl"]
        n_samples = X.shape[0]

        completed_loglikelihood_array = np.zeros(n_sample_trial)
        new_2part_complexity_array = np.zeros(n_sample_trial)
        new_2part_complexity_Z = multinomial_mdl.calculate_mdl(n_samples, self.n_clusters)
        purity = []
        entropy = []

        if true_z is not None:
            true_z = true_z.nonzero()[1] if len(true_z.shape) == 2 else true_z
            NMIs = []

        for i in range(n_sample_trial):
            z = sample_category(self.z_posterior)
            loglikelihood, n_z = _log_likelihood_mle(X, z, if_ss=True)
            completed_loglikelihood_array[i] = loglikelihood
            # print(np.sum([int(nz1) * int(nz2) for nz1 in n_z1 for nz2 in n_z2 if int(nz1) > 0 and int(nz2) > 0]))
            new_2part_complexity_array[i] = new_2part_complexity_Z + np.sum(
                [multinomial_mdl.calculate_mdl(int(nz1) * int(nz2), 2) for nz1 in n_z for nz2 in n_z if
                 int(nz1) > 0 and int(nz2) > 0])
            purity.append(calculate_purity(X, z, self.n_clusters))
            entropy.append(calculate_entropy(X, z, self.n_clusters))

            if true_z is not None:
                NMIs.append(normalized_mutual_info_score(true_z, z))

        old_mdl_rissanen_complexity = self.calculate_NML_penalty(n_samples)

        bic_num = np.log(n_samples)

        index_pos = (X == POSITIVE_VALUE)
        index_neg = (X == NEGATICE_VALUE)
        index_pos_for_z1 = index_pos.astype(np.int8)
        index_neg_for_z1 = index_neg.astype(np.int8)

        complexity = {
            'old_rissanen_complete': (-completed_loglikelihood_array + old_mdl_rissanen_complexity).tolist(),
            'old_rissanen_penalty': [old_mdl_rissanen_complexity],

            'new_em_complete': (-completed_loglikelihood_array + new_2part_complexity_array).tolist(),
            'new_em_penalty': new_2part_complexity_array.tolist(),

            'aic_complete': (-completed_loglikelihood_array + self._n_parameters()).tolist(),
            'aic_penalty': [self._n_parameters()],

            'bic_complete': (-completed_loglikelihood_array + self._n_parameters() * bic_num).tolist(),
            'bic_penalty': [self._n_parameters() * bic_num],

            'purity': np.array(purity),
            'entropy': entropy,
        }

        if true_z is not None:
            complexity["NMI"] = - np.array(NMIs)

        return complexity

    def calculate_NML_penalty(self, n_samples):
        return self._n_parameters() * (np.log(n_samples) * 2 - np.log(2 * np.pi)) / 2 \
               - self.n_clusters * np.log(2) / 2 \
               + self.n_clusters * gammaln((self.n_clusters + 2) / 2) \
               - gammaln(self.n_clusters * (self.n_clusters + 2) / 2) \
               + self.n_clusters * (self.n_clusters + 1) / 2 * np.log(np.pi)


class SBM_model_selector(object):
    def __init__(self, cluster_range_max, verbosity=0):
        self.cluster_range_max = cluster_range_max
        self.verbosity = verbosity

    def fit(self, X):
        numpy2ri.activate()
        X_r = np.array(X)
        sbm = blockmodels.BM_bernoulli(membership_type="SBM", adj=X_r,
                                       verbosity=self.verbosity,
                                       exploration_factor=1.5,
                                       explore_min=self.cluster_range_max,
                                       explore_max=self.cluster_range_max)
        estimate = dollar(sbm, "estimate")
        estimate()

        self.ICL = np.array(dollar(sbm, "ICL"))
        self.learners = []
        for k in range(self.cluster_range_max):
            n_clusters = k + 1
            eta = np.array(dollar(dollar(sbm, "model_parameters")[k], "pi"))
            assert eta.shape == (n_clusters, n_clusters)
            z_posterior = np.array(dollar(dollar(sbm, "memberships")[k], "Z"))
            pi = np.sum(z_posterior, axis=0) + 10 * EPS
            pi /= np.sum(pi)
            learner = TrainedSBM(pi, eta, z_posterior)
            self.learners.append(learner)
        numpy2ri.deactivate()

    def score_new(self, X, calculators, y=None, true_z=None, n_sample_trial=5):
        scores = {(learner.n_clusters,): learner.score_new(X, calculators, true_z=true_z, n_sample_trial=n_sample_trial) \
                  for learner in self.learners}
        methods = scores[(self.learners[0].n_clusters,)].keys()
        result = {method: {cluster: criteria[method] for cluster, criteria in scores.items()} for method in methods}
        result["ICL"] = dict()
        for learner, icl in zip(self.learners, self.ICL):
            result["ICL"][(learner.n_clusters,)] = [[-icl]]
        return result
