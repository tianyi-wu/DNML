from time import time
from collections import Counter

import numpy as np
import scipy.misc as scm
from scipy.special import gammaln
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import logsumexp
from sklearn.utils.validation import check_is_fitted

from ..topic_model.LDA.VB.LatentDirichletAllocationWithSample import loglikelihood as multinomial_mle_log_likelihood

EPS = np.finfo(float).eps
SMALL_THRESHOLD = -1.0e100


def calculate_purity_from_array(arr):
    return - max(Counter(arr).values())


def calculate_entropy_from_array(arr):
    counts = np.array(list(Counter(arr).values()))
    total = np.sum(counts)
    return - np.sum(counts * np.log(counts)) + total * np.log(total)


def calculate_cluster_criteron(X, func):
    return np.array([func(X[:, d]) for d in range(X.shape[1])])


def calculate_purity(X, Z, n_clusters):
    return np.sum([calculate_cluster_criteron(X[Z == k], calculate_purity_from_array) \
                   for k in range(n_clusters) if np.any(Z == k)]) / X.shape[0] / X.shape[1]


def calculate_entropy(X, Z, n_clusters):
    return np.sum([calculate_cluster_criteron(X[Z == k], calculate_entropy_from_array) \
                   for k in range(n_clusters) if np.any(Z == k)]) / X.shape[0] / X.shape[1]


def log_multinomial_density(X, log_thetas):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        _X under each of the n_components multivariate Gaussian distributions.
    """
    lpr = np.zeros((X.shape[0], len(log_thetas)))

    for i in range(X.shape[0]):
        for j in range(len(log_thetas)):
            lpr[i][j] = np.sum([log_thetas[j][dim][value] for dim, value in enumerate(X[i])])
    return lpr


class _MultinomialMixtureModelBase(BaseEstimator):
    """
    Multinormial Mixture Model.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    tol : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold. Defaults to 1e-3.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. The best results is kept.

    verbose : int, default: 0
        Enable verbose output. If 1 then it always prints the current
        initialization and iteration step. If greater than 1 then
        it prints additionally the change and time needed for each step.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    log_thetas_ : list, shape (`n_components`, `n_dimensions`, `n_features`)
        Mean parameters for each mixture component.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    """

    def __init__(self, n_components=1, weight_prior=1, base_prior=1,
                 tol=1e-3, n_iter=100, n_init=1, evaluate_every=10,
                 verbose=0, random_state=None):
        self.n_components = n_components
        self.tol = tol
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.verbose = verbose
        self.evaluate_every = evaluate_every

        self.random_state = random_state
        self.random_state_ = check_random_state(self.random_state)

        # flag to indicate exit status of fit() method: converged (True) or
        # n_iter reached (False)
        self.converged_ = False

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of _X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of _X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in _X.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        check_is_fitted(self, 'log_thetas_')
        X = check_array(X)

        lpr = (log_multinomial_density(X, self.log_thetas_) + self.log_weights_[np.newaxis, :])
        lpr[lpr < SMALL_THRESHOLD] = SMALL_THRESHOLD
        logprob = logsumexp(lpr, axis=1)
        log_responsibilities = lpr - logprob[:, np.newaxis]
        return logprob, log_responsibilities

    def score(self, X, y=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in _X
        """
        logprob, _ = self.score_samples(X)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,) component memberships
        """
        logprob, log_responsibilities = self.score_samples(X)
        return np.exp(log_responsibilities).argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        logprob, log_responsibilities = self.score_samples(X)
        return np.exp(log_responsibilities)

    def perplexity(self, X):
        # return np.exp(-np.mean(self.score(_X)))
        return -np.mean(self.score(X))

    def _init_latent_paras(self, MD):
        init_gamma = 100.
        init_var = 1. / init_gamma
        self.log_thetas_ = [[self.random_state_.gamma(init_gamma, init_var, md) for md in MD] for _ in
                            range(self.n_components)]
        self.log_weights_ = np.log(1.0 / self.n_components) * np.ones(self.n_components)

    def _fit(self, X, MD, y=None, do_prediction=False):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the GMM object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation.
        """

        # initialization step
        X = check_array(X, dtype=np.int64, ensure_min_samples=2, estimator=self)

        n_features = X.shape[1]
        # MD = np.max(_X, axis=0) + 1
        # MD = np.array([int(md) for md in MD])
        initialized = False

        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        if self.n_init is None or self.n_init < 1:
            self.log_thetas_ = [[-np.ones(MD[d]) * np.log(MD[d]) for d in range(n_features)] for _ in
                                range(self.n_components)]
            self.log_weights_ = np.log(1.0 / self.n_components) * np.ones(self.n_components)
            initialized = True
            self.n_init = 1

        evaluate_every = self.evaluate_every
        max_log_prob = -np.infty

        if self.verbose > 0:
            print('Expectation-maximization algorithm started.')

        for init in range(self.n_init):
            if self.verbose > 0:
                print('Initialization ' + str(init + 1))
                start_init_time = time()

            if initialized == False:
                self._init_latent_paras(MD)

            # EM algorithms
            prev_log_likelihood = None
            # reset self.converged_ to False
            self.converged_ = False

            for iteration in range(self.n_iter):
                # Expectation step
                log_likelihoods, log_responsibilities = self.score_samples(X)
                current_log_likelihood = log_likelihoods.mean()

                # Maximization step
                self._do_mstep(X, MD, log_responsibilities)
                if evaluate_every > 0 and iteration % evaluate_every == 0:
                    if self.verbose > 0:
                        print("{0}th iteration".format(iteration))
                    if prev_log_likelihood is not None:
                        # Check for convergence.
                        if current_log_likelihood < prev_log_likelihood:
                            if self.verbose > 0:
                                print("In {2} th iter, log likelihood reversed: old:{0}, new:{1}".format(
                                    prev_log_likelihood,
                                    current_log_likelihood, iteration))
                        change = abs((current_log_likelihood - prev_log_likelihood) / current_log_likelihood)
                        if self.verbose > 1:
                            print('\t\tChange: ' + str(change))
                        if change < self.tol:
                            self.converged_ = True
                            if self.verbose > 0:
                                print('\t\tEM algorithm converged.')
                            break
                    prev_log_likelihood = current_log_likelihood

            # if the results are better, keep it
            if self.n_iter > 1:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
                    best_params = {'log_thetas': self.log_thetas_,
                                   'log_weights_': self.log_weights_}
                    if self.verbose > 1:
                        print('\tBetter parameters were found.')

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter > 1:
            self.log_thetas_ = best_params['log_thetas']
            self.log_weights_ = best_params['log_weights_']

        return np.exp(log_responsibilities)

    def fit(self, X, MD, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the GMM object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self._fit(X, MD, y)
        return self

    def _do_mstep(self, X, MD, log_responsibilities):
        """Perform the Mstep of the EM algorithm and return the cluster weights.
        """
        # for k in range(self.n_components):
        #     for d in range(len(self.log_thetas_[k])):
        #         self.log_thetas_[k][d].fill(EPS)

        for k in range(self.n_components):
            for d in range(X.shape[1]):
                for md in range(MD[d]):
                    mask = (X[:, d] == md)
                    if np.any(mask):
                        self.log_thetas_[k][d][md] = scm.logsumexp(log_responsibilities[mask, k])

        for k in range(self.n_components):
            for d in range(len(self.log_thetas_[k])):
                self.log_thetas_[k][d] = self.log_thetas_[k][d] - scm.logsumexp(self.log_thetas_[k][d])

        self.log_weights_ = np.apply_along_axis(scm.logsumexp, axis=0, arr=log_responsibilities)
        self.log_weights_ -= scm.logsumexp(self.log_weights_)
        return np.exp(self.log_weights_)

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = np.sum([[len(log_theta_1_dim) - 1 for log_theta_1_dim in self.log_thetas_[0]]])
        return ndim * self.n_components + self.n_components - 1


class MultinomialMixtureModel(_MultinomialMixtureModelBase):
    def sample_latent_variable(self, X, log_responsibilities=None, ifSample=True, ifProbabilistic=True):
        rng = self.random_state_
        if log_responsibilities is None:
            _, log_responsibilities = self.score_samples(X)

        n_samples = X.shape[0]
        Z = np.zeros(n_samples, dtype=np.int64)
        n_k = np.zeros(self.n_components, dtype=np.int64)
        n_k_d_v = [[dict() for _ in range(X.shape[1])] for _ in range(self.n_components)]
        if ifSample:
            for n, x in enumerate(X):
                if ifProbabilistic:
                    k = np.nonzero(rng.multinomial(1, np.exp(log_responsibilities[n])))[0][0]
                else:
                    k = np.argmax(log_responsibilities[n])

                Z[n] = k
                for d, xx in enumerate(x):
                    n_k_d_v[k][d][xx] = n_k_d_v[k][d].get(xx, 0) + 1
                n_k[k] += 1

            log_likelihood_Z = -multinomial_mle_log_likelihood(n_k[n_k.nonzero()])

            log_likelihood_X = 0.0
            for n_d_v in n_k_d_v:
                for n_v in n_d_v:
                    log_likelihood_X += -multinomial_mle_log_likelihood(np.array(list(n_v.values()), dtype=np.int64))

            return n_k, log_likelihood_X, log_likelihood_Z, Z
            # np.sum(n_k * self.log_weights_)
        else:
            return np.exp(logsumexp(log_responsibilities, axis=0)), None, None, None

    def score_new(self, X, MD, calculators, y=None, n_sample_trial=10):
        multinomial_mdl = calculators["multinomial_mdl"]
        mixture_multinomial_mdl = calculators["mixture_multinomial_mdl"]

        log_likelihoods, log_responsibilities = self.score_samples(X)

        loglikelihood_X_array = np.zeros(n_sample_trial)
        loglikelihood_Z_array = np.zeros(n_sample_trial)
        new_mdl_array = np.zeros(n_sample_trial)

        new_mdl_SC_Z = multinomial_mdl.calculate_mdl(X.shape[0], self.n_components)
        old_mdl_exact = mixture_multinomial_mdl.calculate_mdl(X.shape[0], self.n_components)
        old_mdl_rissanen = self.rissanen_parametric_complexity(MD, X.shape[0])

        purity_array = np.zeros(n_sample_trial)
        entropy_array = np.zeros(n_sample_trial)
        for idx in range(n_sample_trial):
            n_k, loglikelihood_X, loglikelihood_Z, Z = self.sample_latent_variable(X, log_responsibilities,
                                                                                   ifSample=True,
                                                                                   ifProbabilistic=True)
            # print(n_k, loglikelihood_X, loglikelihood_Z)
            loglikelihood_X_array[idx] = loglikelihood_X
            loglikelihood_Z_array[idx] = loglikelihood_Z

            new_mdl_array[idx] = new_mdl_SC_Z + np.sum(
                [multinomial_mdl.calculate_mdl(int(nk), md) for nk in n_k for md in MD if int(nk) > 0])

            purity_array[idx] = calculate_purity(X, Z, self.n_components)
            entropy_array[idx] = calculate_entropy(X, Z, self.n_components)

        loglikelihood_array = loglikelihood_X_array + loglikelihood_Z_array

        return {'old_exact_complete': (-loglikelihood_array + old_mdl_exact).tolist(),
                'old_exact_penalty': [old_mdl_exact],

                'old_rissanen_complete': (-loglikelihood_array + old_mdl_rissanen).tolist(),
                'old_rissanen_penalty': [old_mdl_rissanen],

                'new_em_complete': (-loglikelihood_array + new_mdl_array).tolist(),
                'new_em_penalty': new_mdl_array.tolist(),

                'aic_complete': (-loglikelihood_array + self._n_parameters()).tolist(),
                'aic_penalty': [self._n_parameters()],

                'bic_complete': (-loglikelihood_array + self._n_parameters() * np.log(X.shape[0]) / 2).tolist(),
                'bic_penalty': [self._n_parameters() * np.log(X.shape[0]) / 2],

                "purity": purity_array,
                "entropy": entropy_array,
                }

    def rissanen_parametric_complexity(self, MD, n_samples):
        MD = np.array(MD)
        K = self.n_components
        D = MD.shape[0]
        # K_base = np.sum(MD) - D
        # K_total = K - 1 + K * K_base
        # logI_base = np.sum((MD - 1) / 2 * np.log(np.pi) + gammaln((MD - 1) / 2))
        return (K - 1 - K * D + K * np.sum(MD)) / 2 * np.log(n_samples / 2) + (K * D - K + 1) / 2 * np.log(np.pi) \
               - gammaln(K * (-D + 1 + np.sum(MD)) / 2) + K * gammaln((-D + 1 + np.sum(MD)) / 2) \
               - K * np.sum(gammaln(MD / 2))
