# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import itertools

from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed

TEST_NAME = "test"


def fit_and_score_normal(learner, X_train, X_test, para_dict, multinomial_mdl, mixture_mdl, n_sample_trial):
    # set parameter
    learner.set_params(**para_dict)
    learner.multinomial_mdl = multinomial_mdl
    learner.mixture_mdl = mixture_mdl

    # fit model
    learner.fit(X_train)

    # calculate criteria
    score = learner.score_new(X_train, n_sample_trial=n_sample_trial)

    if not X_test is None:
        score[TEST_NAME] = [learner.perplexity(X_test)]

    return score, learner


class ModelSelectorAbstract(object):
    __metaclass__ = ABCMeta

    def __init__(self, learner, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        self.learner = learner
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

    @abstractmethod
    def select_model(self, X, para_map, n_trial=5, n_sample_trial=10):
        print('Abstract')


class ModelSelectorMDL(ModelSelectorAbstract):
    def __init__(self, multinomial_mdl, mixture_mdl, learner, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        super(ModelSelectorMDL, self).__init__(learner=learner, n_jobs=n_jobs, verbose=verbose,
                                               pre_dispatch=pre_dispatch)
        self.multinomial_mdl = multinomial_mdl
        self.mixture_mdl = mixture_mdl

    def select_model(self, X_train, para_map, n_trial=5, n_sample_trial=10, X_test=None):
        pre_dispatch = self.pre_dispatch
        para_name = list(para_map.keys())[0]

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=pre_dispatch
        )(
            delayed(fit_and_score_normal)(clone(self.learner), X_train, X_test, {para_name: para}, self.multinomial_mdl,
                                          self.mixture_mdl, n_sample_trial)
            for para in para_map[para_name] for _ in range(n_trial))

        # process output list to result dict
        n_folds = n_trial
        names = out[0][0].keys()

        final_scores = {name: {} for name in names}
        fitted_learners = []

        for index, para in enumerate(para_map[para_name]):
            grid_start = n_folds * index
            scores = {name: [] for name in names}
            for score, learner in out[grid_start:grid_start + n_folds]:
                fitted_learners.append(learner)
                for name, value in score.items():
                    scores[name].append(value)

            for name, value in scores.items():
                final_scores[name][para] = value

        return final_scores, fitted_learners


def fit_and_score(learner, X_train, X_test, fit_argument={}, score_argument={}, perplexity_argument={}):
    learner.fit(X_train, **fit_argument)
    score = learner.score_new(X_train, **score_argument)
    # score = learner.fit_and_score(X_train,  **score_argument)
    # score["model"] = learner
    # calculate perplextiy
    if not X_test is None:
        if isinstance(X_test, list):
            score[TEST_NAME] = [learner.perplexity(X, **perplexity_argument) for X in X_test]
        else:
            score[TEST_NAME] = [learner.perplexity(X_test, **perplexity_argument)]

    return score


def convert_para_map_to_list(para_map):
    names = sorted(para_map.keys())
    return names, list(itertools.product(*[para_map[name] for name in names]))


class ModelSelector(ModelSelectorAbstract):
    def __init__(self, learner, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        super(ModelSelector, self).__init__(learner=learner, n_jobs=n_jobs, verbose=verbose,
                                            pre_dispatch=pre_dispatch)

    def select_model(self, X_train, para_map, n_trial=5, fit_argument={}, score_argument={}, perplexity_argument={}, X_test=None):
        if self.n_jobs == 1:
            return self.select_model_single_process(X_train, para_map, n_trial=n_trial, fit_argument=fit_argument,
                                                    score_argument=score_argument,
                                                    perplexity_argument=perplexity_argument,
                                                    X_test=X_test)
        else:
            return self.select_model_multi_process(X_train, para_map, n_trial=n_trial, fit_argument=fit_argument,
                                                   score_argument=score_argument,
                                                   perplexity_argument=perplexity_argument,
                                                   X_test=X_test)

    def select_model_single_process(self, X_train, para_map, n_trial=5, fit_argument={}, score_argument={}, perplexity_argument={}, X_test=None):
        para_names, para_values_list = convert_para_map_to_list(para_map)

        out = []
        for para_values in para_values_list:
            # learner = clone(self.learner)
            learner = self.learner
            for pn, pv in zip(para_names, para_values):
                # print(pn, pv)
                learner.set_params(**{pn: pv})

            for _ in range(n_trial):
                out.append(fit_and_score(clone(learner), X_train, X_test, fit_argument=fit_argument, score_argument=score_argument, perplexity_argument=perplexity_argument))

        # process output list to result dict
        n_folds = n_trial
        method_names = out[0].keys()

        final_scores = {name: {} for name in method_names}

        for index, para in enumerate(para_values_list):
            grid_start = n_folds * index
            scores = {name: [] for name in method_names}
            for score in out[grid_start:grid_start + n_folds]:
                for name, value in score.items():
                    scores[name].append(value)

            for name, value in scores.items():
                final_scores[name][para] = value

        return final_scores

    def select_model_multi_process(self, X_train, para_map, n_trial=5, fit_argument={}, score_argument={}, perplexity_argument={}, X_test=None):
        pre_dispatch = self.pre_dispatch
        para_name = list(para_map.keys())[0]

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=pre_dispatch
        )(
            delayed(fit_and_score_normal)(clone(self.learner), X_train, X_test, {para_name: para}, self.multinomial_mdl,
                                          self.mixture_mdl, n_sample_trial)
            for para in para_map[para_name] for _ in range(n_trial))

        # process output list to result dict
        n_folds = n_trial
        names = out[0].keys()

        final_scores = {name: {} for name in names}

        for index, para in enumerate(para_map[para_name]):
            grid_start = n_folds * index
            scores = {name: [] for name in names}
            for score in out[grid_start:grid_start + n_folds]:
                for name, value in score.items():
                    scores[name].append(value)

            for name, value in scores.items():
                final_scores[name][para] = value

        return final_scores
