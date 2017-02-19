# -*- coding: utf-8 -*-

import numbers

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

POSITIVE_VALUE = 1
NEGATICE_VALUE = 0
MISSING_VALUE = -1


class SBMDataGenerator(object):
    def __init__(self, n_clusters1, n_clusters2, rho, alpha1, alpha2, beta,
                 n_noise_clusters=0, noise_alpha=None, noise_beta=None,
                 random_state=None):
        self.n_clusters1 = n_clusters1
        self.n_clusters2 = n_clusters2

        if isinstance(alpha1, numbers.Number):
            alpha1 = alpha1 * np.ones(n_clusters1)
        self.alpha1 = alpha1
        if isinstance(alpha2, numbers.Number):
            alpha2 = alpha2 * np.ones(n_clusters2)
        self.alpha2 = alpha2

        self.rho = rho

        self.beta = beta

        self.random_state = random_state
        # if self.random_state is not None:
        #     np.random.set_state(self.random_state)
        self.rng = check_random_state(self.random_state)

        self.eta = self.rho * self.rng.beta(*self.beta, size=(self.n_clusters1, self.n_clusters2))

        self.n_noise_clusters = n_noise_clusters
        if self.n_noise_clusters > 0:
            if noise_beta is None:
                noise_beta = self.beta
            self.noise_eta = self.rng.beta(*noise_beta, size=self.n_noise_clusters)

            if noise_alpha is None:
                noise_alpha = self.alpha1[0]
            self.noise_alpha = noise_alpha * np.ones(self.n_noise_clusters)

    def generate_data(self, n_samples1, n_samples2, train_size=1.0, n_test_matrixs=5, noise_threshold=0.0):
        self.pi1 = self.rng.dirichlet(self.alpha1)
        self.pi2 = self.rng.dirichlet(self.alpha2)

        z1 = np.random.multinomial(1, self.pi1, n_samples1)
        z2 = np.random.multinomial(1, self.pi2, n_samples2)
        X_train = self._generate_data(z1, z2, self.eta)

        idx = np.argwhere(X_train != MISSING_VALUE).T
        _, idx_test = train_test_split(np.array(idx).T, test_size=(1 - train_size))
        X_train[idx_test[:, 0], idx_test[:, 1]] = MISSING_VALUE

        if noise_threshold > 0.0:
            if self.n_noise_clusters == 0:
                raise ValueError
            #
            # noise_index1 = self.rng.choice(range(n_samples1), size=int(noise_threshold*n_samples1), replace=False)
            # for n in noise_index1:
            #     for m in range(n_samples2):
            #         if X_train[n, m] != MISSING_VALUE:
            #             noise_theta = np.random.dirichlet(self.noise_alpha)
            #             k = np.random.multinomial(1, noise_theta).nonzero()[0][0]
            #             X_train[n, m] = np.random.binomial(1, self.noise_eta[k], 1)
            #
            # noise_index2 = self.rng.choice(range(n_samples2), size=int(noise_threshold*n_samples2), replace=False)
            # for m in noise_index2:
            #     for n in range(n_samples1):
            #         if X_train[n, m] != MISSING_VALUE:
            #             noise_theta = np.random.dirichlet(self.noise_alpha)
            #             k = np.random.multinomial(1, noise_theta).nonzero()[0][0]
            #             X_train[n, m] = np.random.binomial(1, self.noise_eta[k], 1)

            nonmissing_sample_indexs = np.argwhere(X_train != -1)
            n_nonmissing_samples = len(nonmissing_sample_indexs)
            noise_index = self.rng.choice(range(n_nonmissing_samples),
                                          size=int(noise_threshold*n_nonmissing_samples), replace=False)
            X_idx_x, X_idx_y = nonmissing_sample_indexs[noise_index].T
            noise_theta = np.random.dirichlet(self.noise_alpha, size=len(X_idx_x))
            for x, y, t in zip(X_idx_x, X_idx_y, noise_theta):
                k = np.random.multinomial(1, t).nonzero()[0][0]
                # print(t, k)
                X_train[x, y] = np.random.binomial(1, self.noise_eta[k], 1)

        X_test = [self._generate_data(z1, z2, self.eta) for _ in range(n_test_matrixs)]
        return X_train, X_test, z1, z2

    @staticmethod
    def _generate_data(z1, z2, eta):
        n_samples1, n_clusters1 = z1.shape
        n_samples2, n_clusters2 = z2.shape
        data = np.zeros([n_samples1, n_samples2], dtype=np.int64)
        for k in range(n_clusters1):
            for l in range(n_clusters2):
                point = (z1[:, k][:, np.newaxis].dot(z2[:, l][np.newaxis, :])).astype(bool)
                data[point] = np.random.binomial(1, eta[k, l], np.sum(point))
        return data

    def sort_data(self, data, z1, z2):
        if len(z1.shape) == 2:
            sort_func = lambda x: np.nonzero(x[1])[0][0]
        else:
            sort_func = lambda x: x[1]
        sorted_data = data
        sorted_data = np.array(
            [row[0] for row in sorted(zip(sorted_data, z1), key=sort_func)])
        sorted_data = np.array(
            [row[0] for row in sorted(zip(sorted_data.T, z2), key=sort_func)]).T
        return sorted_data


class SingleSBMDataGenerator(object):
    def __init__(self, n_clusters, rho, alpha, beta,
                 n_noise_clusters=0, noise_alpha=None, noise_beta=None,
                 random_state=None):
        self.n_clusters = n_clusters

        if isinstance(alpha, numbers.Number):
            alpha = alpha * np.ones(n_clusters)
        self.alpha = alpha
        self.rho = rho
        self.beta = beta

        self.random_state = random_state
        # if self.random_state is not None:
        #     np.random.set_state(self.random_state)
        self.rng = check_random_state(self.random_state)

        self.eta = self.rho * self.rng.beta(*self.beta, size=(self.n_clusters, self.n_clusters))

        self.n_noise_clusters = n_noise_clusters
        if self.n_noise_clusters > 0:
            if noise_beta is None:
                noise_beta = self.beta
            self.noise_eta = self.rng.beta(*noise_beta, size=self.n_noise_clusters)

            if noise_alpha is None:
                noise_alpha = self.alpha[0]
            self.noise_alpha = noise_alpha * np.ones(self.n_noise_clusters)

    def generate_data(self, n_samples, train_size=1.0, n_test_matrixs=1, noise_threshold=0.0):
        self.pi = self.rng.dirichlet(self.alpha)

        z = np.random.multinomial(1, self.pi, n_samples)
        X_train = self._generate_data(z, self.eta)

        idx = np.argwhere(X_train != MISSING_VALUE).T
        _, idx_test = train_test_split(np.array(idx).T, test_size=(1 - train_size))
        X_train[idx_test[:, 0], idx_test[:, 1]] = MISSING_VALUE

        if noise_threshold > 0.0:
            if self.n_noise_clusters == 0:
                raise ValueError

            nonmissing_sample_indexs = np.argwhere(X_train != -1)
            n_nonmissing_samples = len(nonmissing_sample_indexs)
            noise_index = self.rng.choice(range(n_nonmissing_samples),
                                          size=int(noise_threshold*n_nonmissing_samples), replace=False)
            X_idx_x, X_idx_y = nonmissing_sample_indexs[noise_index].T
            noise_theta = np.random.dirichlet(self.noise_alpha, size=len(X_idx_x))
            for x, y, t in zip(X_idx_x, X_idx_y, noise_theta):
                k = np.random.multinomial(1, t).nonzero()[0][0]
                # print(t, k)
                X_train[x, y] = np.random.binomial(1, self.noise_eta[k], 1)

        X_test = [self._generate_data(z, self.eta) for _ in range(n_test_matrixs)]
        return X_train, X_test, z

    @staticmethod
    def _generate_data(z, eta):
        n_samples, n_clusters = z.shape
        data = np.zeros([n_samples, n_samples], dtype=np.int64)
        for k in range(n_clusters):
            for l in range(n_clusters):
                point = (z[:, k][:, np.newaxis].dot(z[:, l][np.newaxis, :])).astype(bool)
                data[point] = np.random.binomial(1, eta[k, l], np.sum(point))
        return data

    def sort_data(self, data, z):
        if len(z1.shape) == 2:
            sort_func = lambda x: np.nonzero(x[1])[0][0]
        else:
            sort_func = lambda x: x[1]

        sorted_data = data
        order = [x[0] for x in sorted(enumerate(z), key=sort_func, reverse=True)]
        return sorted_data[order, :][:, order]


class MMSBMDataGenerator(object):
    def __init__(self, n_clusters, rho, alpha, beta,
                 n_noise_clusters=0, noise_alpha=None, noise_beta=None,
                 random_state=None):
        self.random_state = random_state
        self.rng = check_random_state(self.random_state)
        self.n_clusters = n_clusters
        self.rho = rho

        if isinstance(alpha, numbers.Number):
            alpha = alpha * np.ones(n_clusters)
        self.alpha = alpha
        self.beta = beta
        self.eta = rho * self.rng.beta(*self.beta, size=(self.n_clusters, self.n_clusters))

        self.n_noise_clusters = n_noise_clusters
        if self.n_noise_clusters > 0:
            if noise_beta is None:
                noise_beta = self.beta
            self.noise_eta = self.rng.beta(*noise_beta, size=self.n_noise_clusters)

            if noise_alpha is None:
                noise_alpha = self.alpha[0]
            self.noise_alpha = noise_alpha * np.ones(self.n_noise_clusters)

    def generate_data(self, n_samples, train_size=1.0, n_test_matrixs=5, noise_threshold=0.0):
        self.theta = np.random.dirichlet(self.alpha, size=n_samples)
        X_train = self._generate_data(self.theta, self.eta)
        idx = np.argwhere(X_train != MISSING_VALUE).T

        _, idx_test = train_test_split(np.array(idx).T, test_size=(1 - train_size))
        X_train[idx_test[:, 0], idx_test[:, 1]] = MISSING_VALUE

        if noise_threshold > 0.0:
            if self.n_noise_clusters == 0:
                raise ValueError
            X_idx_x, X_idx_y = np.argwhere(np.random.random(X_train.shape) < noise_threshold).T
            noise_theta = np.random.dirichlet(self.noise_alpha, size=len(X_idx_x))
            for x, y, t in zip(X_idx_x, X_idx_y, noise_theta):
                k = np.random.multinomial(1, t).nonzero()[0][0]
                X_train[x, y] = np.random.binomial(1, self.noise_eta[k], 1)

        X_test = [self._generate_data(self.theta, self.eta) for _ in range(n_test_matrixs)]
        return X_train, X_test

    @staticmethod
    def _generate_data(theta, eta):
        n_samples = theta.shape[0]
        data = np.zeros([n_samples, n_samples], dtype=np.int64)
        for n in range(n_samples):
            for m in range(n_samples):
                if n == m:
                    data[n, m] = MISSING_VALUE
                else:
                    k = np.random.multinomial(1, theta[n]).nonzero()[0][0]
                    l = np.random.multinomial(1, theta[m]).nonzero()[0][0]
                    data[n, m] = np.random.binomial(1, eta[k, l])
        return data

    def sort_data(self, data, theta):
        pca = PCA(n_components=1, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto',
                  random_state=None)
        compressed_theta = pca.fit_transform(theta[:, :-1])
        print(pca.explained_variance_ratio_)

        order = [o for o, s in sorted(enumerate(compressed_theta[:, 0]), key=lambda x: x[1])]
        sorted_data = data[order, :][:, order]
        return sorted_data
