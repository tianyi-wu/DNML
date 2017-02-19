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
maptpx = importr("maptpx")
base = importr("base")
dollar = base.__dict__["$"]


class LDA_R(BaseEstimator, TransformerMixin):
    def __init__(self, n_topics=5, verbose=0):
        self.n_topics = n_topics
        self.verbose = verbose

    def fit(self, X):
        numpy2ri.activate()
        X_r = np.array(X)
        lda = maptpx.topics(X_r, K=self.n_topics, verb=self.verbose)
        self.doc_topic_distr = np.array(dollar(lda, "omega"))
        self.components_ = np.array(dollar(lda, "theta")).T
        numpy2ri.deactivate()

    def select(self, X, topic_range):
        if len(topic_range) == 1:
            if topic_range[0] == 2:
                topic_range_ = np.array([2, 3])
            else:
                topic_range_ = np.array([2, topic_range[0]])
        else:
            topic_range_ = topic_range
        numpy2ri.activate()
        X_r = np.array(X)
        lda = maptpx.topics(X_r, K=np.array(topic_range_), verb=self.verbose)
        criteria = np.array(dollar(lda, "BF"))
        numpy2ri.deactivate()

        if len(topic_range) == 1:
            if topic_range[0] == 2:
                return np.array([criteria[0]])
            else:
                return np.array([criteria[1]])
        else:
            return np.array(criteria)

