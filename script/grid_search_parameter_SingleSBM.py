# -*- coding: utf-8 -*-
import argparse
import copy
import pickle
import rpy2
import os
import sys
import subprocess
origin_stdout = sys.stdout

import numpy as np
import pandas as pd

from ..model.MDL.MDL import MultiNomialMDL
from ..model.relation_model.DataGenerater import SingleSBMDataGenerator
from ..model.relation_model.SBM.SBM_R import SBM_model_selector

opt_file = \
"""
--loops={loops}			# length of search
--nchains={nchains}			# run one chain
--temp={temp}			# temperature parameter for MC^3
--mcmcflag={mcmcflag}			# run hill-climbing, not MCMC
--outroot={outroot}		# location for output
--configfile={configfile}	# location of config file
--graphname={graphname}	# location of relation file
--hypupdates={hypupdates}			# try updating hyperparameters 5 times per
                #    iteration
--betamagupdate={betamagupdate}		# update betamag	(see below)
--betapropupdate={betapropupdate}		# don't update betaprop (see below)
"""

configure_file = \
"""
irm1

1 1

{n_samples} {n_samples} 1 1

2 0 0
"""

relation = "0 {n} {m} {e} \n"

IRM_path = "/home/wty36/Dropbox/programming/research/pycharm/SBM/irm/irm"

RE_RUN_NUMBER = 3


def run_subprocess(cmd):
    for iteration in range(RE_RUN_NUMBER):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = p.communicate()
        if p.returncode == 0:
            return
        else:
            print(stdout_data.decode("utf-8"), stderr_data.decode("utf-8"))
    raise SystemError


def cal_data_to_score(n_clusters, rho, alpha, beta, n_samples, max_para,
                      n_noise_clusters, noise_threshold,
                      n_score_sample_trials, calculators, root_directory, index, **kwargs):
    noise_alpha = 0.05
    noise_beta = (1,  1)
    train_size = 1.0
    dg = SingleSBMDataGenerator(n_clusters, rho, alpha, beta,
                                n_noise_clusters=n_noise_clusters, noise_alpha=noise_alpha, noise_beta=noise_beta,
                                random_state=None)
    X_train, _, z = dg.generate_data(n_samples, train_size=train_size, n_test_matrixs=0,
                                          noise_threshold=noise_threshold)

    # IRM
    loops = 100  # length of search
    nchains = 1  # run one chain
    temp = 5  # temperature parameter for MC^3
    mcmcflag = 0  # run hill-climbing, not MCMC
    # initfile= # can specify initial class assignments if desired.
    hypupdates = 5 # try updating hyperparameters 5 times per iteration
    betamagupdate = 1  # update betamag	(see below)
    betapropupdate = 0  # don't update betaprop (see below)
    outroot = os.path.join(root_directory, "irm{0}_".format(index))  # location for output
    configfile = os.path.join(root_directory, "irm{0}_.config".format(index))
    graphname = os.path.join(root_directory, "irm{0}_.graph".format(index))
    optfile = os.path.join(root_directory, "irm{0}_.opt".format(index))
    stat_file = os.path.join(root_directory, "irm{}_stat".format(index))
    _base = {name: value for name, value in locals().items() if
             name[0] != "_" and type(value) in [float, int, str, dict, list]}
    with open(optfile, "w") as f:
        _body = opt_file.format(**_base)
        f.write(_body)
    with open(configfile, "w") as f:
        _body = configure_file.format(**_base)
        f.write(_body)
    with open(graphname, "w") as f:
        for n in range(n_samples):
            for m in range(n_samples):
                if n != m:
                    f.write(relation.format(n=n, m=m, e=X_train[n, m]))
    cmd = '{0} @{1} > {2}'.format(IRM_path, optfile, stat_file)
    print(cmd)
    try:
        run_subprocess(cmd)
        numbers = list(pd.read_csv(stat_file, sep='\s+', header=None)[3])
    except SystemError:
        numbers = []

    cmd = 'rm {0}/irm{1}_*'.format(root_directory,index)
    try:
        run_subprocess(cmd)
    except SystemError:
        pass

    score = {"irm": [numbers]}
    ms = SBM_model_selector(cluster_range_max=max_para, verbosity=0)

    try:
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            ms.fit(X_train)
    finally:
        sys.stdout = origin_stdout

    score = ms.score_new(X_train, calculators, y=None, true_z=z, n_sample_trial=n_score_sample_trials)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is to calculate score')
    parser.add_argument("-conf", "--configure", required=True, help="pickled configure file")
    args = parser.parse_args()

    with open(args.configure, "rb") as f:
        configure = pickle.load(f)

    mdl_path = configure["mdl_path"]
    with open(mdl_path, "rb") as f:
        multinomial_mdl = pickle.load(f)

    calculators = {"multinomial_mdl": multinomial_mdl}

    n_trial = configure["n_trial"]
    n_samples_array = configure["n_samples_array"]

    results = []
    for n in n_samples_array:
        new_conf = copy.deepcopy(configure)
        new_conf["n_samples"] = n
        result = {"base": new_conf,
                  "result": []}
        results.append(result)

        for trial_index in range(n_trial):
            try:
                score = cal_data_to_score(calculators=calculators, **new_conf)
                result["result"].append(score)
            except rpy2.rinterface.RRuntimeError as e:
                print(e)

        with open(configure["output_path"] + "irm", "wb") as f:
            pickle.dump(results, f)
