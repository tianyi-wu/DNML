# -*- coding: utf-8 -*-

from abc import ABCMeta

import numpy as np
import scipy.misc as scm
from numpy.random import dirichlet
from scipy.misc import logsumexp
from scipy.special import gammaln, multigammaln

EPS = np.finfo(np.float).eps


class MDL(object):
    __metaclass__ = ABCMeta
    #
    # @abstractmethod
    # def calculate_mdl(self, n, k):
    #     print('Abstract')


class MultiNomialMDL(MDL):
    def __init__(self):
        self.n_k_sc = {}

    def calculate_mdl(self, n, k, threshold=5000) -> float:
        if n == 0:
            raise ValueError
        elif n == 1:
            return np.log(k)
        elif n > threshold:
            return self.rissanen_approximate(n, k)
        elif n in self.n_k_sc:
            k_sc_list = self.n_k_sc[n]
            if len(k_sc_list) - 1 >= k:
                return k_sc_list[k]
            else:
                new_k_sc_list = np.append(k_sc_list, np.zeros(3 * (k - (len(k_sc_list) - 1))))
                for i in range(len(k_sc_list), len(new_k_sc_list)):
                    new_k_sc_list[i] = scm.logsumexp(
                        [new_k_sc_list[i - 1], new_k_sc_list[i - 2] + np.log(n) - np.log(i - 2)])
                self.n_k_sc[n] = new_k_sc_list
                return new_k_sc_list[k]
        else:
            k_sc_list = np.zeros(max(4, k + 1))
            k_sc_list[1] = 0
            entropy = np.array([0.] + [hi * np.log(hi) - gammaln(hi + 1) for hi in range(1, n + 1)])
            k_sc_list[2] = scm.logsumexp(entropy + entropy[::-1]) + gammaln(n + 1) - n * np.log(n)
            for i in range(3, max(4, k + 1)):
                k_sc_list[i] = scm.logsumexp([k_sc_list[i - 1], k_sc_list[i - 2] + np.log(n) - np.log(i - 2)])
            self.n_k_sc[n] = k_sc_list
            return k_sc_list[k]

    @staticmethod
    def rissanen_approximate(n, k):
        return (k - 1) / 2.0 * (np.log(n) - np.log(2 * np.pi)) + k / 2.0 * np.log(np.pi) - gammaln(k / 2.0)


class LDAMDLRissanen(MDL):
    def __init__(self, D, V, random_state=None, sample=5000):
        self.D = D
        self.V = V
        self.k_integral = dict()

        self.random_state = random_state
        self.sample = sample

    def _calculate_one_sample(self, alpha):
        log_theta = np.log(dirichlet(alpha, size=self.D))
        return np.sum(np.apply_along_axis(scm.logsumexp, 0, log_theta) * self.V / 2.0) - (np.sum(log_theta) / 2.0)

    def calculate_mdl(self, n_data, K, n_free_para=None):
        if n_free_para is None:
            raise ValueError

        if K in self.k_integral:
            log_fisher_matrix_int = self.k_integral[K]
        else:
            alpha = np.ones(K)
            sampled_results = np.array([self._calculate_one_sample(alpha) for _ in range(self.sample)])
            log_fisher_matrix_int = scm.logsumexp(sampled_results) - np.log(self.sample)
            self.k_integral[K] = log_fisher_matrix_int
        # print(((K * (self.V_sqrt - 1) + self.D * (K - 1)) / 2.0 * (np.log(n_data) - np.log(2.0 * np.pi)), \
        #        K * (self.V_sqrt * np.log(np.pi) / 2.0 - gammaln(self.V_sqrt / 2.0)), \
        #        log_fisher_matrix_int))
        return K * (self.V * np.log(np.pi) / 2.0 - gammaln(self.V / 2.0)) + log_fisher_matrix_int \
               + n_free_para / 2.0 * (np.log(n_data) - np.log(2.0 * np.pi))

class MMMMDL(MDL):
    def __init__(self, multinomial_mdl, N_max):
        self.MD = None
        self.N_max = N_max
        self.multinomial_mdl = multinomial_mdl

        self.coefficient = np.zeros(self.N_max + 1)
        for n in range(1, self.N_max + 1):
            self.coefficient[n] = n * np.log(n) - gammaln(n + 1)

        self.n_dim_n_complexity_n_data_ = {}
        self.data_number_map = None

    def set_MD(self, MD):
        if MD != self.MD:
            self.MD = MD
            if self.MD not in self.n_dim_n_complexity_n_data_:
                self.n_dim_n_complexity_n_data_[self.MD] = dict()
                self.data_number_map = self.n_dim_n_complexity_n_data_[self.MD]
                self.initialize_MD()
            else:
                self.data_number_map = self.n_dim_n_complexity_n_data_[self.MD]

    def initialize_MD(self):
        self.data_number_map[1] = np.zeros(self.N_max + 1)
        for n in range(1, self.N_max + 1):
            self.data_number_map[1][n] = np.sum([self.multinomial_mdl.calculate_mdl(n, d) for d in self.MD])
        self.data_number_map[1] += self.coefficient

    def calculate_mdl(self, n_samples, n_clusters):
        if n_clusters == 1:
            if n_samples < len(self.data_number_map[1]):
                return self.data_number_map[1][n_samples] - self.coefficient[n_samples]
            else:
                return np.sum([self.multinomial_mdl.calculate_mdl(n_samples, d) for d in self.MD])

        if n_samples == 1:
            return self.data_number_map[1][n_samples] + np.log(n_clusters) - self.coefficient[n_samples]

        if n_clusters in self.data_number_map:
            return self.data_number_map[n_clusters][n_samples] - self.coefficient[n_samples]

        m = int(np.log2(n_clusters))

        for i in range(1, m + 1):
            m_power = 2 ** i
            if m_power not in self.data_number_map:
                self.data_number_map[m_power] = np.zeros(self.N_max + 1)
                self.data_number_map[m_power][1] = self.data_number_map[1][1] + np.log(m_power)

                for n in range(2, self.N_max + 1):
                    self.data_number_map[m_power][n] = scm.logsumexp(
                        self.data_number_map[m_power / 2][1:n] + self.data_number_map[m_power / 2][1:n][::-1])

        nonzeor_index = np.nonzero(list(map(int, bin(n_clusters)[2:][::-1])))[0]
        # print(nonzeor_index)
        index = 2 ** nonzeor_index[0]
        C_mmm = self.data_number_map[index]
        k_cum = index
        for i in nonzeor_index[1:]:
            index = 2 ** i
            k_cum += index
            # print(index, k_cum)
            if k_cum not in self.data_number_map:
                self.data_number_map[k_cum] = np.zeros(self.N_max + 1)
                self.data_number_map[k_cum][1] = self.data_number_map[1][1] + np.log(k_cum)

                for n in range(2, self.N_max + 1):
                    self.data_number_map[k_cum][n] = scm.logsumexp(C_mmm[1:n] + self.data_number_map[index][1:n][::-1])
            C_mmm = self.data_number_map[k_cum]
        return C_mmm[n_samples] - self.coefficient[n_samples]

    def calculate_mdl_range(self, array, K):
        self.calculate_mdl(np.max(array), K)
        return self.data_number_map[K][array] - self.coefficient[array]

    # def calculate_mdl_fft(self, N_max, K):
    #     m = int(np.log2(K))
    #     self.sc = np.zeros((N_max, m + 1))
    #
    #     coefficient = np.array([(n + 1) * np.log(n + 1) - gammaln(n + 2) for n in range(N_max)])
    #     #print(coefficient)
    #
    #     for n in range(N_max):
    #         self.sc[n][0] = np.sum([self.mnmdl.calculate_mdl(n + 1, d) for d in self.MD])
    #     self.sc[:, 0] += coefficient
    #
    #     for i in range(1, m + 1):
    #         self.sc[0, i] = self.sc[0][0] + np.log(2 ** i) + coefficient[0]
    #         old = np.exp(self.sc[:, (i - 1)])[:-1]
    #         self.sc[1:, i] = np.log(np.convolve(old, old)[:(N_max-1)])
    #
    #     nonzeor_index = np.nonzero(list(map(int, bin(K)[2:][::-1])))[0]
    #     if len(nonzeor_index) == 1:
    #         return self.sc[:, nonzeor_index[0]] - coefficient
    #
    #     print(nonzeor_index)
    #     result = np.zeros(N_max)
    #     result[0] = self.sc[0][0] + np.log(K) + coefficient[0]
    #     result[1:] = np.log(
    #         np.convolve(np.exp(self.sc[:, nonzeor_index[0]])[:-1], np.exp(self.sc[:, nonzeor_index[1]])[:-1])[:(N_max-1)])
    #
    #     for i in nonzeor_index[2:]:
    #         result[1:] = np.log(np.convolve(np.exp(result[:1]), np.exp(self.sc[:, i])[:-1])[:(N_max-1)])
    #
    #     return result - coefficient
    #
    #
    #
    # def calculate_mdl_direct(self, N_max, K):
    #     self.sc = np.zeros((N_max, K))
    #     keisu1 = np.array([(n + 1) * np.log(n + 1) - gammaln(n + 2) for n in range(N_max)])
    #     keisu2 = np.array([gammaln(n + 2) - (n + 1) * np.log(n + 1) for n in range(N_max)])
    #
    #     for n in range(N_max):
    #         self.sc[n][0] = np.sum([self.mnmdl.calculate_mdl(n + 1, d) for d in self.MD])
    #
    #     for k in range(1, K):
    #         self.sc[0][k] = self.sc[0][0] + np.log(k)
    #
    #     for k in range(1, K):
    #         for n in range(1, N_max):
    #             nn = n + 1
    #             temp = np.array(
    #                 [self.sc[nnn - 1][0] + self.sc[nn - nnn - 1][k - 1] + keisu1[nnn - 1] + keisu1[nn - nnn - 1] + keisu2[nn - 1]
    #                  for nnn in range(1, nn)])
    #             self.sc[n][k] = scm.logsumexp(temp)
    #
    #     return self.sc

    def calculate_mdl_direct_recursive(self, N, K):
        if N == 1:
            return np.sum([self.multinomial_mdl.calculate_mdl(1, d) for d in self.MD]) + np.log(K)
        elif K == 1:
            return np.sum([self.multinomial_mdl.calculate_mdl(N, d) for d in self.MD])
        else:
            return scm.logsumexp([self.calculate_mdl_direct_recursive(n,
                                                                      int(K / 2)) + self.calculate_mdl_direct_recursive(
                N - n, K - int(K / 2)) + (n) * np.log(n) - gammaln(n + 1) + (N - n) * np.log(N - n) - gammaln(N - n + 1)
                                  for n in range(1, N)]) - (N) * np.log(N) + gammaln(N + 1)


class GaussianMDL(object):
    @staticmethod
    def calculate_mdl(n_data, n_dim, mu, sigma2):
        return - n_dim * mu / 2.0 * np.log(sigma2) \
               + n_dim / 2.0 * ((n_data + mu + 1) * np.log(n_data + mu) \
                                - (n_data + mu) * np.log(2) - (n_data + mu) \
                                - mu * np.log(np.pi) - (mu + 1) * np.log(mu)) \
               + multigammaln(mu / 2.0, n_dim) \
               - multigammaln((n_data + mu) / 2.0, n_dim)


class OldGaussianMDL(object):
    @staticmethod
    def calculate_mdl(n_data, lambda_min, R):
        if n_data == 1:
            return np.log(4) + np.log(R) / 2.0 - np.log(lambda_min) / 2.0 - gammaln(1 / 2) \
                   + n_data / 2.0 * (np.log(n_data) - np.log(2) - 1)
        else:
            return np.log(4) + np.log(R) / 2.0 - np.log(lambda_min) / 2.0 - gammaln(1 / 2) \
                   + n_data / 2.0 * (np.log(n_data) - np.log(2) - 1) - gammaln((n_data - 1) / 2)


class GMMMDL(MDL):
    def __init__(self, N, lambda_min, R):
        self.lambda_min = lambda_min
        self.R = R
        self.N = N
        self.keisu = np.zeros(self.N + 1)
        for n in range(1, self.N + 1):
            self.keisu[n] = n * np.log(n) - gammaln(n + 1)

        self.data_number_map = {}
        self.data_number_map[1] = np.zeros(self.N + 1)
        for n in range(1, self.N + 1):
            self.data_number_map[1][n] = OldGaussianMDL.calculate_mdl(n, lambda_min, R)
        self.data_number_map[1] += self.keisu

    # def __init__(self, N_max, n_dim, mu, sigma2):
    #     self.mu = mu
    #     self.sigma2 = sigma2
    #     self.n_dim = n_dim
    #     self.N_max = N_max
    #     self.coefficient = np.zeros(self.N_max + 1)
    #     for n in range(1, self.N_max + 1):
    #         self.coefficient[n] = n * np.log(n) - gammaln(n + 1)
    #
    #     self.data_number_map = {}
    #     self.data_number_map[1] = np.zeros(self.N_max + 1)
    #     for n in range(1, self.N_max + 1):
    #         self.data_number_map[1][n] = OldGaussianMDL.calculate_mdl(n, self.n_dim, self.mu, self.sigma2)
    #     self.data_number_map[1] += self.coefficient

    def calculate_mdl(self, n_data, K):
        if K in self.data_number_map:
            return self.data_number_map[K][n_data] - self.keisu[n_data]

        m = int(np.log2(K))

        for i in range(1, m + 1):
            k = 2 ** i
            if k not in self.data_number_map:
                self.data_number_map[k] = np.zeros(self.N + 1)
                self.data_number_map[k][1] = self.data_number_map[1][1] + np.log(k)

                for n in range(2, self.N + 1):
                    self.data_number_map[k][n] = scm.logsumexp(
                        self.data_number_map[k / 2][1:n] + self.data_number_map[k / 2][1:n][::-1])

        nonzeor_index = np.nonzero(list(map(int, bin(K)[2:][::-1])))[0]
        index = 2 ** nonzeor_index[0]
        C_mmm = self.data_number_map[index]
        k_cum = index
        for i in nonzeor_index[1:]:
            index = 2 ** i
            k_cum += index
            if k_cum not in self.data_number_map:
                self.data_number_map[k_cum] = np.zeros(self.N + 1)
                self.data_number_map[k_cum][1] = self.data_number_map[1][1] + np.log(k_cum)

                for n in range(2, self.N + 1):
                    self.data_number_map[k_cum][n] = scm.logsumexp(C_mmm[1:n] + self.data_number_map[index][1:n][::-1])
            C_mmm = self.data_number_map[k_cum]

        return C_mmm[n_data] - self.keisu[n_data]


class SNML2MDL(object):
    def __init__(self, M_max=1000):
        self.arr_kaijo = gammaln(np.array(list(range(1, M_max + 2))))

    def extend(self, M):
        old_M = self.arr_kaijo.shape[0] - 1
        if M > old_M:
            self.arr_kaijo = np.append(self.arr_kaijo, gammaln(np.array(list(range(old_M + 2, M * 2)))))

    def calculate_2SNML(self, M, n_k):
        self.extend(M)
        n_k = np.array(n_k) + EPS
        N = np.sum(n_k)
        result = None

        if M == 1:
            result = np.sum(n_k * np.log(n_k)) + logsumexp(np.log(n_k + 1.0) + n_k * np.log(1.0 + 1 / n_k))
        elif M == 0 or len(n_k) == 1:
            result = np.sum(n_k * np.log(n_k))
        else:
            middle = int(len(n_k) / 2)
            b1 = self._calculate_2SNML(M, n_k, 0, middle - 1) - self.arr_kaijo[:(M + 1)]
            b2 = self._calculate_2SNML(M, n_k, middle, len(n_k) - 1)[::-1] - self.arr_kaijo[M::-1]
            result = self.arr_kaijo[M] + logsumexp(b1 + b2)
        return result - (N + M) * np.log(N + M)

    def _calculate_2SNML(self, M, n_k, i, j):
        # print(i,j)
        if i == j:
            return np.array([(m + n_k[i]) * np.log(m + n_k[i]) for m in range(M + 1)])
        elif i < j:
            middle = int((i + j) / 2) + 1
            b1 = self._calculate_2SNML(M, n_k, i, middle - 1) - self.arr_kaijo[:(M + 1)]
            b2 = self._calculate_2SNML(M, n_k, middle, j) - self.arr_kaijo[:(M + 1)]
            b = self.arr_kaijo[:(M + 1)].copy()
            b[0] += np.sum(n_k[i:(j + 1)] * np.log(n_k[i:(j + 1)]))
            b[1:] += np.array([logsumexp(b1[0:(m + 1)] + b2[m::-1]) for m in range(1, M + 1)])
            return b
        else:
            raise ValueError

    def calculate_2SNML_recursive(self, M, n_k):
        N = np.sum(n_k)
        return self._calculate_2SNML_recursive(M, n_k) - (N + M) * np.log(N + M)

    def _calculate_2SNML_recursive(self, M, n_k):
        result = None
        if M == 1:
            result = np.sum(n_k * np.log(n_k)) + logsumexp(np.log(n_k + 1.0) + n_k * np.log(1.0 + 1 / n_k))
        elif M == 0:
            result = np.sum(n_k * np.log(n_k))
        elif len(n_k) == 1:
            result = (n_k[0] + M) * np.log(n_k[0] + M)
        else:
            middle = int(len(n_k) / 2)
            left = n_k[:middle]
            right = n_k[middle:]
            b = np.zeros(M + 1)
            for r1 in range(M + 1):
                r2 = M - r1
                b[r1] = gammaln(M + 1) - gammaln(r1 + 1) - gammaln(r2 + 1) + \
                        self._calculate_2SNML_recursive(r1, left) + \
                        self._calculate_2SNML_recursive(r2, right)
            result = logsumexp(b)
        return result
