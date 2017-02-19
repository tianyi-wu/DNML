# -*- coding: utf-8 -*-
import argparse
import copy
import os
import pickle

import numpy as np
from rpy2.rinterface import RRuntimeError
from sklearn.base import clone
from sklearn.model_selection import KFold

from ..model.MDL.MDL import MultiNomialMDL, LDAMDLRissanen
from ..model_selector.ModelSelector import ModelSelectorMDL

from ..model.topic_model.LDA.LDA_R import LDA_R
from ..model.topic_model.LDA.VB.LatentDirichletAllocationWithScore import LatentDirichletAllocationWithScore

COHERENCE_NAME = "coherence"
COHERENCE_MATRIX_NAME = "coherence_matrix"
TOPIC_WORD_DISTRIBUTION_NAME = "topic_word"
DOC_TOPIC_DISTRIBUTION_NAME = "doc_topic"
HDP_PATH = "DOWNLOAD HDP FROM BLEI LAB"


def cal_data_to_score(X_train, X_test,
                      multinomial_mdl, mixture_mdl,
                      para_map,
                      n_jobs=1, n_trial=1, n_sample_trial=10, verbose=0,
                      **kwargs):
    # VB
    learner = LatentDirichletAllocationWithScore(verbose=0, learning_method='batch', evaluate_every=50, perp_tol=1e-3,
                                                 max_iter=6000)
    selector = ModelSelectorMDL(multinomial_mdl, mixture_mdl, learner, n_jobs=n_jobs, verbose=verbose)
    scores_VB, fitted_learners = selector.select_model(X_train, para_map=para_map, n_trial=1,
                                                       n_sample_trial=n_sample_trial,
                                                       X_test=X_test)

    para_name = list(para_map.keys())[0]
    para_values = sorted(para_map[para_name])

    # laplace
    learner = LDA_R(verbose=0)
    try:
        logBFs = -learner.select(X_train, topic_range=para_values)
    except RRuntimeError:
        logBFs = [np.inf for _ in para_values]
    scores_VB["laplace_approx"] = {para_value:[[logBF]] for para_value, logBF in zip(para_values, logBFs)}

    return scores_VB


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is to calculate score')
    parser.add_argument("-conf", "--configure", required=True, help="pickled configure file")
    args = parser.parse_args()

    with open(args.configure, "rb") as f:
        configure = pickle.load(f)

    data_path = configure["data_path"]
    with open(data_path, "rb") as f:
        X_train, X_test = pickle.load(f)

    D = X_train.shape[0]
    V = X_train.shape[1]
    mixture_mdl = LDAMDLRissanen(D, V, random_state=None, sample=5000)
    mdl_path = configure["mdl_path"]
    with open(mdl_path, "rb") as f:
        multinomial_mdl = pickle.load(f)

    n_trials = configure["n_trials"]
    scores = []
    for _ in range(n_trials):
        scores.append(cal_data_to_score(X_train=X_train, X_test=X_test,
                              multinomial_mdl=multinomial_mdl, mixture_mdl=mixture_mdl,
                              **configure))

    if os.path.isfile(configure["output_path"]):
        with open(configure["output_path"], "rb") as f:
            old_socres = pickle.load(f)
        scores.extend(old_socres)
        with open(configure["output_path"], "wb") as f:
            pickle.dump({"base": configure, "result": scores}, f)
    else:
        with open(configure["output_path"], "wb") as f:
            pickle.dump({"base": configure, "result": scores}, f)
