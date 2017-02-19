# -*- coding: utf-8 -*-
import argparse
import copy
import pickle

from sklearn.base import clone
from sklearn.model_selection import KFold

from ..model.MDL.MDL import MultiNomialMDL, MMMMDL
from ..model.MMM.DummyMMM import DummyMMM
from ..model.MMM.MultinomialMixtureModel import MultinomialMixtureModel
from ..model_selector.ModelSelector import ModelSelector


def cal_data_to_score(n_components, MD, alpha, beta, n_samples, para_map,
                      n_score_sample_trials, n_jobs, calculators, **kwargs):
    dd = DummyMMM(k=n_components, MD=MD, alpha=alpha, beta=beta)
    X_train = dd.generate_artificial_data(n_samples, noise_threshold=0.0)
    X_test = dd.generate_artificial_data(500, noise_threshold=0.0)

    learner = MultinomialMixtureModel(n_components=3, random_state=None, tol=1e-6, n_iter=800, evaluate_every=10,
                                      n_init=5, verbose=0)
    ms = ModelSelector(learner)
    result = ms.select_model(X_train, para_map, n_trial=1, fit_argument={"MD": MD},
                             score_argument={"calculators": calculators,
                                             "n_sample_trial": n_score_sample_trials,
                                             "MD": MD}, X_test=X_test)

    learner = MultinomialMixtureModel(n_components=3, random_state=None, tol=1e-3, n_iter=500, evaluate_every=20,
                                      n_init=5, verbose=0)
    para_name = list(para_map.keys())[0]
    para_values = sorted(para_map[para_name])
    kf = KFold(n_splits=5)
    scores_perplexity = {}
    for para_value in para_values:
        perplexities = []
        learner_temp = clone(learner)
        learner_temp.set_params(**{para_name: para_value})
        for train, test in kf.split(X_train):
            learner_temp.fit(X_train[train], MD=MD)
            perplexities.append(learner_temp.perplexity(X_train[test]))
        scores_perplexity[para_value] = [perplexities]
    result.update({'cv': scores_perplexity})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is to calculate score')
    parser.add_argument("-conf", "--configure", required=True, help="pickled configure file")
    args = parser.parse_args()

    with open(args.configure, "rb") as f:
        configure = pickle.load(f)

    mdl_path = configure["mdl_path"]
    if len(mdl_path) == 0:
        multinomial_mdl = MultiNomialMDL()
    else:
        with open(mdl_path, "rb") as f:
            multinomial_mdl = pickle.load(f)

    n_sample_array = configure["n_sample_array"]
    MD = configure["MD"]
    mixture_multinomial_mdl = MMMMDL(multinomial_mdl, n_sample_array[-1])
    mixture_multinomial_mdl.set_MD(MD)

    calculators = {"multinomial_mdl": multinomial_mdl,
                   "mixture_multinomial_mdl": mixture_multinomial_mdl,
                   }

    n_trial = configure["n_trial"]

    results = []
    for n_index, n_samples in enumerate(n_sample_array):
        new_conf = copy.deepcopy(configure)
        new_conf["n_samples"] = n_samples
        result = {"base": new_conf,
                  "result": []}
        results.append(result)
        for trial_index in range(n_trial):
            score = cal_data_to_score(calculators=calculators, **new_conf)
            result["result"].append(score)

        with open(configure["output_path"], "wb") as f:
            pickle.dump(results, f)
