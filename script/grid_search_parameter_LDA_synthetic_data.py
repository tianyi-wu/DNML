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

from ..model.topic_model.ArtificialDataGenerator import LDAArtificialDataGenerator
from ..model.topic_model.LDA.LDA_R import LDA_R
from ..model.topic_model.LDA.VB.LatentDirichletAllocationWithScore import LatentDirichletAllocationWithScore

from ..model.topic_model.HDP.blei_hdp_c_util import write_data_to_file

COHERENCE_NAME = "coherence"
COHERENCE_MATRIX_NAME = "coherence_matrix"
TOPIC_WORD_DISTRIBUTION_NAME = "topic_word"
DOC_TOPIC_DISTRIBUTION_NAME = "doc_topic"
HDP_PATH = "DOWNLOAD HDP FROM BLEI LAB"


def create_data(dd, noise_dd, D, N, n_noise_docs):
    # X = dd.generate_artificial_data(int(D * (1 - noise_threshold)), N, noise_threshold=0.0)
    # X_noise = noise_dd.generate_artificial_data(int(D * noise_threshold), N, noise_threshold=0.0)
    X = dd.generate_artificial_data(D - n_noise_docs, N, noise_threshold=0.0)
    X_noise = noise_dd.generate_artificial_data(n_noise_docs, N, noise_threshold=0.0)
    X = np.vstack((X, X_noise))
    np.random.shuffle(X)
    return X


def cal_data_to_score(K, D, N, V, alpha, beta,
                      n_noise_topics, noise_ratio, para_map, n_jobs, n_sample_trial,
                      verbose, **kwargs):
    # noise_threshold = noise_ratio * n_noise_topics
    # noise_threshold = noise_ratio
    n_noise_docs = noise_ratio

    dd = LDAArtificialDataGenerator(K, V, alpha, beta, k_noise=0, noise_alpha=1, noise_beta=0.1,
                                    random_state=None)
    noise_dd = LDAArtificialDataGenerator(n_noise_topics, V, alpha=0.05, beta=0.05, k_noise=0, noise_alpha=1,
                                          noise_beta=0.1,
                                          random_state=None)

    X_train = create_data(dd, noise_dd, D, N, n_noise_docs)
    X_test = create_data(dd, noise_dd, 1000, 500, n_noise_docs)

    # VB
    learner = LatentDirichletAllocationWithScore(verbose=0, learning_method='batch', evaluate_every=50,
                                                 perp_tol=1e-3,
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
    scores_VB["laplace_approx"] = {para_value: [[logBF]] for para_value, logBF in zip(para_values, logBFs)}

    return scores_VB


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is to calculate score')
    parser.add_argument("-conf", "--configure", required=True, help="pickled configure file")
    args = parser.parse_args()

    with open(args.configure, "rb") as f:
        configure = pickle.load(f)

    K = configure["K"]
    D = configure["D"]
    V = configure["V"]
    n_trial = configure["n_trial"]

    if_run_hdp = configure["if_run_hdp"]
    qsub_directory = os.path.join(configure["root_directory"], str(configure["index"]))

    mixture_mdl = LDAMDLRissanen(D, V, random_state=None, sample=5000)
    mdl_path = configure["mdl_path"]
    if len(mdl_path) == 0:
        multinomial_mdl = MultiNomialMDL()
    else:
        with open(mdl_path, "rb") as f:
            multinomial_mdl = pickle.load(f)

    n_sample_array = configure["n_sample_array"]

    results = []
    for n_index, n in enumerate(n_sample_array):
        new_conf = copy.deepcopy(configure)
        new_conf["N"] = n
        result = {"base": new_conf,
                  "result": []}
        results.append(result)

        if if_run_hdp:
            qsub_n_directory = os.path.join(qsub_directory, str(n_index + 1))
            if not os.path.exists(qsub_n_directory):
                os.makedirs(qsub_n_directory)

    for trial_index in range(n_trial):
        for n_index, n in enumerate(n_sample_array):
            new_conf = results[n_index]["base"]
            score, (X_train, X_test) = cal_data_to_score(**new_conf)
            results[n_index]["result"].append(score)

            if if_run_hdp:
                qsub_n_directory = os.path.join(qsub_directory, str(n_index + 1))
                qsub_n_trial_directory = os.path.join(qsub_n_directory, str(trial_index + 1))
                if not os.path.exists(qsub_n_trial_directory):
                    os.makedirs(qsub_n_trial_directory)
                write_data_to_file(X_train, X_test, qsub_n_trial_directory)

                # try:
                #     hdp_train_cmd = "{0} --train_data {1}/train.dat --directory {1} --save_lag -1 --max_iter 1000 > /dev/null".format(HDP_PATH, qsub_n_trial_directory)
                #     run_subprocess(hdp_train_cmd, RE_RUN_NUMBER=5)
                #     hdp_test_cmd = "{0} --test_data {1}/test.dat --model_prefix {1}/final --directory {1} --save_lag -1  --max_iter 500 > /dev/null".format(HDP_PATH, qsub_n_trial_directory)
                #     run_subprocess(hdp_test_cmd, RE_RUN_NUMBER=5)
                # except SystemError:
                #     pass
                # finally:
                #     clean_cmd = "find {output}/$i/ -type f -not -name '*.json' -not -name '*.log' -print0 | xargs -0 rm --"
                #     run_subprocess(clean_cmd, RE_RUN_NUMBER=1)

        with open(configure["output_path"], "wb") as f:
            pickle.dump(results, f)
