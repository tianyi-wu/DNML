# -*- coding: utf-8 -*-
import itertools
import os
import pickle
import shutil

import numpy as np

_run_qsub_script = """for textfile in $( ls . | grep .sub$ ); do qsub "${textfile}"; done;"""

_qsub_script = """
#!/bin/bash
#PBS -l nodes=1:ppn={n_jobs}:taqy
#PBS -o ./log_{index}.out
#PBS -e ./log_{index}.err
#PBS -N {name}

export MKL_NUM_THREADS={n_jobs}
export OMP_NUM_THREADS={n_jobs}

python ~/Dropbox/programming/research/pycharm/MDL/grid_search_parameter_SingleSBM.py -conf {configure_path}
"""

_mdl_path = os.path.join(os.path.dirname(__file__), "../../dataset/pre_calculated_multinomial_mdl.pkl")
_result_path = os.path.join(os.path.dirname(__file__), "../../temp")

if __name__ == '__main__':
    n_clusters = 10
    _n_samples = 500
    _n_samples_array = np.logspace(np.log10(100), np.log10(_n_samples ** 2), num=10)

    n_samples_array = [int(np.sqrt(n)) for n in _n_samples_array]

    rho = 1.0
    alpha = 3
    beta = np.array([1.0, 1.0])

    max_para = 14
    n_noise_clusters = 3

    n_score_sample_trials = 5
    n_jobs = 1
    n_trial = 5
    root_directory = os.path.join(_result_path, "grid_search_SingleSBM_{0}".format(n_clusters))
    mdl_path = _mdl_path

    if not os.path.exists(root_directory):
        os.makedirs(root_directory)
    else:
        for _file in [os.path.join(root_directory, name) for name in os.listdir(root_directory)]:
            if os.path.isfile(_file):
                os.remove(_file)
            else:
                shutil.rmtree(_file)
    index = 0
    _hyperparameters = {
        "rho": [1.0, 0.75, 0.5],
        "alpha": [1, 4],
        "single_beta": [0.1, 0.3, 0.6, 1, 3],
        "noise_threshold": [0.0, 0.01, 0.03, 0.06],
        # "n_noise_clusters": [1, 3],
    }


    _names = sorted(_hyperparameters.keys())
    for values in list(itertools.product(*[_hyperparameters[name] for name in _names])):
        _hyperparameter = {name: value for name, value in zip(_names, values)}
        _hyperparameter["beta"] = beta * _hyperparameter["single_beta"]

        configure_path = os.path.join(root_directory, "{0}.conf".format(index))
        output_path = os.path.join(root_directory, "{0}.npz".format(index))
        name = "grid_search_parameter_SingleSBM_{0}".format(index)
        _base = {name: value for name, value in locals().items() if
                 name[0] != "_" and type(value) in [float, int, str, dict, list]}

        _base.update(_hyperparameter)

        with open(configure_path, "wb") as f:
            pickle.dump(_base, f)

        with open(os.path.join(root_directory, "{0}.sub".format(index)), "w") as f:
            _body = _qsub_script.format(**_base)
            f.write(_body)

        index += 1

        print(_base)

    with open(os.path.join(root_directory, "run_qsub.sh"), "w") as f:
        f.write(_run_qsub_script)
