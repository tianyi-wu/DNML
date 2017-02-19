# -*- coding: utf-8 -*-
from collections import Counter
import itertools
import numbers

import numpy as np
from numpy.random import dirichlet
from sklearn.utils import check_random_state


def split_matrix_by_col(X, test_size):
    X_train = X.copy()
    X_test = np.zeros(X.shape, dtype=np.int64)
    for doc_index in range(X_train.shape[0]):
        word_vec = X_train[doc_index]
        word_list = list(
            itertools.chain.from_iterable(
                [[word_index] * word_vec[word_index] for word_index in word_vec.nonzero()[0]]))
        c = Counter(np.random.choice(word_list, size=int(len(word_list) * test_size), replace=False))
        for word_index, word_count in c.items():
            X_train[doc_index][word_index] -= word_count
            X_test[doc_index][word_index] += word_count
    return X_train, X_test


def get_alpha(D):
    # alpha = 5.0 / D
    alpha = 0.1
    return alpha


def get_beta(V):
    # beta = 18.0 / V_sqrt
    beta = 0.1
    return beta


def _unique_word_number(N, D, K):
    return max(80, int(np.sqrt(N * D)))
    # return max(80, int(np.sqrt(N_max * D) * max(np.log(K), 2)))


def unique_word_number(N, D, K=5):
    V = _unique_word_number(N, D, K)
    # print(V_sqrt)
    alpha, beta = get_alpha(D), get_beta(V)
    # alpha, beta = 5.0 / D, 18.0 / V_sqrt

    dd = LDAArtificialDataGenerator(K, V, alpha, beta)
    X = dd.generate_artificial_data(D, N)
    # _X = corpus2csc(corpus).T.toarray()
    v = np.sum(X, 0)
    # print(v)
    return len(np.nonzero(v)[0])


class ArtificialDataGenerator(object):
    pass


class LDAArtificialDataGenerator(object):
    def __init__(self, k, V, alpha, beta, k_noise=0, noise_alpha=1, noise_beta=0.5, random_state=None):
        self.k = k
        self.V = V
        if isinstance(alpha, numbers.Real):
            self.alpha = alpha * np.ones(self.k)
        else:
            self.alpha = alpha
        assert len(self.alpha) == self.k
        if isinstance(beta, numbers.Real):
            self.beta = beta * np.ones(self.V)
        else:
            self.beta = beta
        assert len(self.beta) == self.V

        self.random_state = random_state
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        self.rng = check_random_state(self.random_state)
        self.Phi = dirichlet(self.beta, size=self.k)
        self.Theta = None
        self.Z = None

        if k_noise > 0:
            self.k_noise = k_noise
            self.noise_alpha = noise_alpha * np.ones(self.k_noise)
            self.noise_beta = noise_beta * np.ones(self.V)
            Phi_noise = dirichlet(self.noise_beta, size=self.k_noise)
            self.Phi = np.vstack((self.Phi, Phi_noise))

    def generate_artificial_data(self, D, N, noise_threshold=0.0):
        N = int(N)
        self.Theta = dirichlet(self.alpha, size=D)
        if noise_threshold > 0.0:
            Theta_noise = dirichlet(self.noise_alpha, size=D)
            self.Theta = np.hstack((self.Theta * (1 - noise_threshold), Theta_noise * noise_threshold))

        self.Z = np.array([np.random.multinomial(N, self.Theta[i]) for i in range(D)])
        X = np.zeros((D, self.V), dtype=np.int)
        for d, counts in enumerate(self.Z):
            for k in range(len(counts)):
                if counts[k] > 0:
                    X[d] += np.random.multinomial(counts[k], self.Phi[k])
        return X

    def _sampler_word(self, counts):
        words = []
        for i in range(len(counts)):
            if counts[i] > 0:
                wordCounts = np.random.multinomial(counts[i], self.Phi[i])
                words.extend([(i, c) for i, c in enumerate(wordCounts) if c > 0])
        return words


class PAMArtificialDataGenerator(object):
    def __init__(self, n_super_topics, n_sub_topics, V, super_alpha, sub_alpha, beta, n_sub_noises=0, alpha_noise=1,
                 beta_noise=0.5, random_state=None):
        self.n_super_topics = n_super_topics
        self.n_sub_topics = n_sub_topics
        self.V = V

        self.super_dirichlet_parameter = super_alpha * np.ones(self.n_super_topics)
        self.super_sub_dirichlet_parameter = sub_alpha * np.ones(self.n_sub_topics)
        self.sub_word_dirichlet_parameter = beta * np.ones(self.V)

        self.random_state = random_state
        self.rng = check_random_state(self.random_state)
        self.sub_word_parameter = dirichlet(self.sub_word_dirichlet_parameter, size=self.n_sub_topics)
        self.Theta = None
        self.doc_super = None
        self.doc_sub = None

        if n_sub_noises > 0:
            self.n_sub_noises = n_sub_noises
            self.super_sub_noise_dirichlet_parameter = alpha_noise * np.ones(self.n_sub_noises)
            self.sub_word_noise_dirichlet_parameter = beta_noise * np.ones(self.V)
            sub_word_parameter_noise = dirichlet(self.sub_word_noise_dirichlet_parameter, size=self.n_sub_noises)
            self.sub_word_parameter = np.vstack((self.sub_word_parameter, sub_word_parameter_noise))

    def generate_artificial_data(self, D, N, noise_threshold=0.0):
        N = int(N)
        self.super_Theta = dirichlet(self.super_dirichlet_parameter, size=D)
        self.sub_Theta = dirichlet(self.super_sub_dirichlet_parameter, size=(D, self.n_super_topics))
        if noise_threshold > 0.0:
            sub_Theta_noise = dirichlet(self.super_sub_noise_dirichlet_parameter, size=(D, self.n_super_topics))
            self.sub_Theta = np.concatenate((self.sub_Theta * (1 - noise_threshold), sub_Theta_noise * noise_threshold),
                                            axis=2)

        self.doc_super = np.array([np.random.multinomial(N, self.super_Theta[i]) for i in range(D)])
        self.doc_sub = np.zeros((D, self.sub_Theta.shape[2]))
        for d, super_counts in enumerate(self.doc_super):
            for super_topic, num in enumerate(super_counts):
                if num > 0:
                    self.doc_sub[d] += np.random.multinomial(num, self.sub_Theta[d][super_topic])

        X = np.zeros((D, self.V), dtype=np.int)
        for d, sub_counts in enumerate(self.doc_sub):
            for sub_topic, num in enumerate(sub_counts):
                if num > 0:
                    X[d] += np.random.multinomial(num, self.sub_word_parameter[sub_topic])
        return X


class PAMArtificialDataGeneratorFromWeiLi(object):
    def __init__(self, n_super_topics, n_sub_topics, V, super_sub_structure, super_alpha, sub_alpha, beta,
                 n_sub_topic_noises=0, alpha_noise=0.8, beta_noise=0.5, random_state=None):
        self.n_super_topics = n_super_topics
        self.n_sub_topics = n_sub_topics
        self.V = V
        self.V_sqrt = int(np.sqrt(V))
        assert self.V_sqrt ** 2 == V
        self.super_sub_structure = super_sub_structure

        self.super_dirichlet_parameter = super_alpha * np.ones(self.n_super_topics)
        self.sub_dirichlet_parameter = sub_alpha * np.ones(self.n_sub_topics)
        self.sub_word_dirichlet_parameter = beta * np.ones(self.V_sqrt)

        self.random_state = random_state
        self.rng = check_random_state(self.random_state)
        self.sub_word_parameter = self._initialize_sub_word_parameter()

        self.Theta = None
        self.doc_super_sample = None
        self.doc_sub_sample = None

        if n_sub_topic_noises >= 0:
            self.n_sub_topic_noises = n_sub_topic_noises
            self.alpha_noise, self.beta_noise = alpha_noise, beta_noise
            if n_sub_topic_noises > 0:
                self.__initialize_sub_word_noise_parameter()
                # self._initialize_sub_word_noise_parameter()

    def _initialize_sub_word_noise_parameter(self):
        super_word_parameter = np.array(
            [np.mean(self.sub_word_parameter[sub_topics], axis=0) for sub_topics in
             self.super_sub_structure.values()])
        self.sub_word_parameter = np.vstack((self.sub_word_parameter, super_word_parameter))

    def __initialize_sub_word_noise_parameter(self):
        self.sub_word_noise_dirichlet_parameter = self.beta_noise * np.ones(self.V)
        sub_word_parameter_noise = dirichlet(self.sub_word_noise_dirichlet_parameter, size=self.n_sub_topic_noises)
        self.sub_word_parameter = np.vstack((self.sub_word_parameter, sub_word_parameter_noise))

    def _initialize_sub_word_parameter(self):
        sub_word_parameter = np.zeros((self.n_sub_topics, self.V))
        self.V_grid = np.array(range(self.V)).reshape(self.V_sqrt, self.V_sqrt)
        dim, index = None, None
        sampled = {(dim, index)}
        for sub_topic in range(self.n_sub_topics):
            while (dim, index) in sampled:
                dim, index = int(self.rng.rand() + 0.5), self.rng.randint(0, self.V_sqrt)
            sampled.add((dim, index))
            words = self.V_grid[index, :] if dim < 0.5 else self.V_grid[:, index]
            sub_word_parameter[sub_topic, words] = dirichlet(self.sub_word_dirichlet_parameter)
        return sub_word_parameter

    def generate_artificial_data(self, D, N, noise_threshold=0.0):
        N = int(N)
        self.doc_super_prob = dirichlet(self.super_dirichlet_parameter, size=D)
        self.doc_super_sub_prob = self._sample_doc_super_sub_prob(D)

        if noise_threshold >= 0.0:
            if self.n_sub_topic_noises > 0:
                self.sub_noise_dirichlet_parameter = self.alpha_noise * np.ones(self.n_sub_topic_noises)
                doc_super_sub_noise_prob = dirichlet(self.sub_noise_dirichlet_parameter, size=(D, self.n_super_topics))

                self.doc_super_sub_prob = np.concatenate(
                    (self.doc_super_sub_prob * (1 - noise_threshold), doc_super_sub_noise_prob * noise_threshold),
                    axis=2)

            self.doc_super_sub_prob = np.concatenate(
                (self.doc_super_sub_prob * (1 - noise_threshold),
                 np.ones((D, self.n_super_topics, 1)) * noise_threshold), axis=2)

        self.doc_super_sample = np.array([np.random.multinomial(N, self.doc_super_prob[i]) for i in range(D)])
        self.doc_sub_sample = np.zeros((D, self.doc_super_sub_prob.shape[2]))
        for d, super_counts in enumerate(self.doc_super_sample):
            for super_topic, num in enumerate(super_counts):
                if num > 0:
                    self.doc_sub_sample[d] += np.random.multinomial(num, self.doc_super_sub_prob[d][super_topic])

        X = np.zeros((D, self.V), dtype=np.int)
        for d, sub_counts in enumerate(self.doc_sub_sample):
            for sub_topic, num in enumerate(sub_counts):
                if num > 0:
                    X[d] += np.random.multinomial(num, self.sub_word_parameter[sub_topic])
        return X

    def _sample_doc_super_sub_prob(self, D):
        doc_super_sub_prob = np.zeros((D, self.n_super_topics, self.n_sub_topics))
        for super_topic in range(self.n_super_topics):
            sub_topics = self.super_sub_structure[super_topic]
            doc_super_sub_prob[:, super_topic, sub_topics] = dirichlet(
                self.sub_dirichlet_parameter[self.super_sub_structure[super_topic]], size=D)
        return doc_super_sub_prob


class hPAMArtificialDataGenerator(object):
    def __init__(self, n_super_topics, n_sub_topics, V, super_alpha, sub_alpha, beta, super_topic_balance=1.0,
                 sub_topic_balance=1.0, n_sub_noises=0, alpha_noise=1, beta_noise=0.5, random_state=None):
        self.n_super_topics = n_super_topics
        self.n_sub_topics = n_sub_topics
        self.V = V

        self.super_dirichlet_parameter = super_alpha * np.ones(self.n_super_topics + 1)
        self.super_dirichlet_parameter[self.n_super_topics] *= super_topic_balance
        self.super_sub_dirichlet_parameter = sub_alpha * np.ones((self.n_super_topics, self.n_sub_topics + 1))
        self.super_sub_dirichlet_parameter[:, self.n_sub_topics] = sub_topic_balance * sub_alpha
        sub_word_dirichlet_parameter = beta * np.ones(self.V)

        self.random_state = random_state
        self.rng = check_random_state(self.random_state)
        root_word_prob = dirichlet(sub_word_dirichlet_parameter)
        super_word_prob = dirichlet(sub_word_dirichlet_parameter, size=self.n_super_topics)
        sub_word_prob = dirichlet(sub_word_dirichlet_parameter, size=self.n_sub_topics)
        self.ROOT_INDEX = [0]
        self.SUPER_INDEX = list(range(1, self.n_super_topics + 1))
        self.SUB_INDEX = list(range(self.n_super_topics + 1, 1 + self.n_super_topics + self.n_sub_topics))
        self.topic_word_prob = np.vstack((root_word_prob, super_word_prob, sub_word_prob))

        self.Theta = None
        self.doc_super = None
        self.doc_topic = None

        if n_sub_noises > 0:
            self.n_sub_noises = n_sub_noises
            self.topic_noise_dirichlet_parameter = alpha_noise * np.ones(self.n_sub_noises)
            self.word_noise_dirichlet_parameter = beta_noise * np.ones(self.V)
            self.noise_word_prob = dirichlet(self.word_noise_dirichlet_parameter, size=self.n_sub_noises)

    def generate_artificial_data(self, D, N, noise_threshold=0.0):
        self.super_Theta = dirichlet(self.super_dirichlet_parameter, size=D)
        self.sub_Theta = np.array([dirichlet(sub_dirichlet_parameter, size=D) for sub_dirichlet_parameter in
                                   self.super_sub_dirichlet_parameter])
        self.sub_Theta = np.swapaxes(self.sub_Theta, 0, 1)

        if noise_threshold > 0.0:
            N_noise = int(noise_threshold * N)
            N = N - N_noise
            X_noise = self.generate_noise_data(D, N_noise)

        self.doc_super = np.array([np.random.multinomial(N, self.super_Theta[i]) for i in range(D)])
        self.doc_topic = np.zeros((D, self.topic_word_prob.shape[0]))

        for d, super_counts in enumerate(self.doc_super):
            for super_topic, num in enumerate(super_counts):
                if num > 0:
                    if super_topic == self.n_super_topics:
                        self.doc_topic[d][self.ROOT_INDEX] += num
                    else:
                        sub_topic = np.random.multinomial(num, self.sub_Theta[d][super_topic])
                        if sub_topic[self.n_sub_topics] > 0:
                            self.doc_topic[d][self.SUPER_INDEX[super_topic]] += sub_topic[self.n_sub_topics]
                        self.doc_topic[d][self.SUB_INDEX] += sub_topic[:self.n_sub_topics]

        X = np.zeros((D, self.V), dtype=np.int)
        for d, topic_counts in enumerate(self.doc_topic):
            for sub_topic, num in enumerate(topic_counts):
                if num > 0:
                    X[d] += np.random.multinomial(num, self.topic_word_prob[sub_topic])

        if noise_threshold > 0.0:
            return X + X_noise
        else:
            return X

    def generate_noise_data(self, D, N):
        X = np.zeros((D, self.V), dtype=np.int)
        noise_Theta = dirichlet(self.topic_noise_dirichlet_parameter, size=D)
        for d, noise_d_theta in enumerate(noise_Theta):
            noise_count = np.random.multinomial(N, noise_d_theta)
            for noise, num in enumerate(noise_count):
                X[d] += np.random.multinomial(num, self.noise_word_prob[noise])
        return X


class hPAMArtificialDataGeneratorWithStructure(hPAMArtificialDataGenerator):
    def __init__(self, n_super_topics, n_sub_topics, super_sub_structure, V, super_alpha, sub_alpha, beta,
                 super_topic_balance=1.0, sub_topic_balance=1.0, n_sub_noises=0, alpha_noise=1, beta_noise=0.5,
                 word_usage_ratio=0.9, word_random_ratio=0.2, random_state=None):
        super(hPAMArtificialDataGeneratorWithStructure, self).__init__(n_super_topics, n_sub_topics, V,
                                                                       super_alpha, sub_alpha, beta,
                                                                       super_topic_balance=super_topic_balance,
                                                                       sub_topic_balance=sub_topic_balance,
                                                                       n_sub_noises=n_sub_noises,
                                                                       alpha_noise=alpha_noise, beta_noise=beta_noise,
                                                                       random_state=random_state)
        self.super_sub_structure = super_sub_structure
        self.super_dirichlet_parameter = super_alpha * np.ones(self.n_super_topics + 1)
        self.super_dirichlet_parameter[self.n_super_topics] *= super_topic_balance
        self.super_sub_dirichlet_parameter = np.zeros((self.n_super_topics, self.n_sub_topics + 1))
        for super_topic, sub_dirichlet_parameter in enumerate(self.super_sub_dirichlet_parameter):
            sub_dirichlet_parameter[self.super_sub_structure[super_topic]] = sub_alpha
        self.super_sub_dirichlet_parameter[:, self.n_sub_topics] = sub_topic_balance * sub_alpha

        shuffled_word_indexes = np.array(range(V))
        np.random.shuffle(shuffled_word_indexes)

        shuffled_word_indexes_for_random = np.array(range(V))
        np.random.shuffle(shuffled_word_indexes_for_random)

        n_total_topics = 1 + self.n_super_topics + self.n_sub_topics
        n_words_per_topic = int(word_usage_ratio / n_total_topics * V)
        n_random_words_per_topic = int(word_random_ratio / n_total_topics * V)

        self.topic_word_prob = np.zeros((n_total_topics, V))
        for i in range(n_total_topics):
            word_dirichlet_parameter = np.zeros(V)
            word_index = list(set(np.hstack((shuffled_word_indexes[
                                             (i * n_words_per_topic):((i + 1) * n_words_per_topic)],
                                             shuffled_word_indexes_for_random[
                                             (i * n_random_words_per_topic):((i + 1) * n_random_words_per_topic)]))))
            word_dirichlet_parameter[word_index] = 1.0
            self.topic_word_prob[i] = dirichlet(word_dirichlet_parameter)
