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
python ~/Dropbox/programming/research/pycharm/MDL/grid_search_parameter_MMM.py -conf {configure_path}
"""

_mdl_path = os.path.join(os.path.dirname(__file__), "../../dataset/pre_calculated_multinomial_mdl.pkl")
_result_path = os.path.join(os.path.dirname(__file__), "../../temp")


if __name__ == '__main__':
    n_components = 5
    _n_samples = 1000
    n_sample_array = [int(n) for n in np.logspace(np.log10(10), np.log10(_n_samples), num=8)]

    para_map = {"n_components": range(2, 9, 1)}

    n_score_sample_trials = 6
    n_jobs = 1
    n_trial = 8
    root_directory = os.path.join(_result_path, "grid_search_MMM_{0}_more".format(n_components))
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
        "alpha":  [1, 5],
        "beta": [0.05, 0.15, 0.3, 0.5],
        "MD": [(16, 16, 16),
               (6, 6, 6, 6),
               (12, 12, 12, 12),
               (8, 8, 8, 8, 8),
               (12, 12, 12, 12, 12),
               (4, 4, 4, 4, 4, 4),
               (6, 6, 6, 6, 6, 6),
               ]
    }

    names = list(_hyperparameters.keys())
    for values in list(itertools.product(*[_hyperparameters[name] for name in names])):
        _hyperparameter = {name: value for name, value in zip(names, values)}

        configure_path = os.path.join(root_directory, "{0}.conf".format(index))
        output_path = os.path.join(root_directory, "{0}.npz".format(index))
        name = "grid_search_parameter_MMM_{0}".format(index)
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
