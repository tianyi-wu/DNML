# -*- coding: utf-8 -*-
import numbers

import numpy as np
from scipy.stats import dirichlet, entropy
from sklearn.utils import check_random_state


def preprocess_input_1_dim(X):
    count = 0
    map = {}
    for x in sorted(X):
        if x not in map:
            map[x] = count
            count += 1
    new_X = np.zeros(len(X))
    for i, x in enumerate(X):
        new_X[i] = map[x]
    return new_X


def preprocess_input(X):
    new_X = np.zeros(X.shape)
    n_features = X.shape[1]
    for i in range(n_features):
        new_X[:, i] = preprocess_input_1_dim(X[:, i])
    return np.array([[int(xx) for xx in x] for x in new_X])


def MultinomialKL(theta1, theta2):
    distance = 100
    for i in range(len(theta1)):
        distance = min(distance, entropy(theta1[i], theta2[i]))
    return distance


class DummyMMM(object):
    def __init__(self, k, MD, alpha, beta, random_state=None):
        self.k = k
        if isinstance(alpha, numbers.Real):
            self.alpha = alpha * np.ones(self.k)
        else:
            self.alpha = alpha
        assert len(self.alpha) == self.k

        self.MD = MD
        if isinstance(beta, numbers.Real):
            self.betas = [[beta * np.ones(md) for md in self.MD] for _ in range(self.k)]
        else:
            self.betas = beta
        for kk in range(self.k):
            for d, md in enumerate(MD):
                assert len(self.betas[kk][d]) == md

        self.random_state = random_state
        self.weights = dirichlet.rvs(self.alpha, size=1, random_state=self.random_state)[0]
        self.thetas = [
            [dirichlet.rvs(beta_1_dim, size=1, random_state=self.random_state)[0] for beta_1_dim in self.betas[kk]] for
            kk in range(self.k)]

        # self.thetas = [[dirichlet.rvs(beta_1_dim, size=1, random_state=self.random_state)[0] for beta_1_dim in self.betas[0]]]
        # for kk in range(self.k-1):
        #     for _ in range(10):
        #         [dirichlet.rvs(beta_1_dim, size=1, random_state=self.random_state)[0] for beta_1_dim in self.betas[kk]]
        #     pass

    def generate_artificial_data(self, n, noise_threshold=0.0):
        rng = check_random_state(self.random_state)
        n = int(n)
        self.Z = np.array([np.nonzero(rng.multinomial(1, self.weights))[0][0] for _ in range(n)])
        self.X = np.zeros((n, len(self.betas[0])))

        for i in range(n):
            theta = self.thetas[self.Z[i]]
            noises = rng.rand(len(self.MD))
            noises_candidate = rng.rand(len(self.MD))
            self.X[i] = np.zeros(len(self.MD))
            for j, theta_1_dim in enumerate(theta):
                if noises[j] < noise_threshold:
                    self.X[i][j] = int(noises_candidate[j] * self.MD[j])
                else:
                    self.X[i][j] = np.nonzero(np.random.multinomial(1, theta_1_dim))[0][0]
        self.X = preprocess_input(self.X)
        return self.X
