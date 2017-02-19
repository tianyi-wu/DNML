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

python ~/Dropbox/programming/research/pycharm/MDL/grid_search_parameter_LDA_synthetic_data.py -conf {configure_path}

for step in {{1..{steps}}}
do
    for trial in {{1..{n_trial}}}
    do
        {HDP_PATH}  --train_data {root_directory}/{index}/$step/$trial/train.dat --sample_hyper --directory {root_directory}/{index}/$step/$trial --save_lag -1 --max_iter 2000 > /dev/null
        {HDP_PATH} --test_data {root_directory}/{index}/$step/$trial/test.dat --model_prefix {root_directory}/{index}/$step/$trial/final --directory {root_directory}/{index}/$step/$trial --save_lag -1  --max_iter 1000 > /dev/null
        find {root_directory}/{index}/$step/$trial/ -type f ! -name '*.log' ! -name 'final.topics' -print0 | xargs -0 rm --
    done
done
"""

HDP_PATH = "/home/wty36/Dropbox/programming/research/pycharm/MDL/hdp/hdp-faster/hdp"
_mdl_path = os.path.join(os.path.dirname(__file__), "../../dataset/pre_calculated_multinomial_mdl.pkl")
_result_path = os.path.join(os.path.dirname(__file__), "../../temp")

if __name__ == '__main__':
    K = 5
    # D = 600
    _N_start = 5
    _N_end = 400
    steps = 10
    n_sample_array = [int(n) for n in np.logspace(np.log10(_N_start), np.log10(_N_end), num=steps)]

    # V = 300
    # alpha = 0.1
    # beta = 0.1
    n_noise_topics = 0
    # noise_threshold = 0.0
    noise_ratio = 0

    _topics = range(1, 10, 1)
    para_map = {"n_topics": _topics}

    n_jobs = 1
    n_trial = 5
    n_sample_trial = 1
    verbose = 0

    if_run_hdp = 1

    # root_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../result/grid_search_LDA_noise/")
    root_directory = os.path.join(_result_path, "grid_search_LDA_complete_K_5")
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
        "V": [200, 400, 600],
        "alpha": [0.1, 0.2, 0.25, 0.3, 0.35],
        "beta": [0.1, 0.2, 0.25, 0.3, 0.35],
        "D": [500, 800]
        # [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    # _hyperparameters = {
    #     "n_noise_topics": list(range(0, 61, 2)),
    # }

    # names = sorted(_hyperparameters.keys(), reverse=True)
    names = ["D", "V", "alpha", "beta"]
    for values in list(itertools.product(*[_hyperparameters[name] for name in names])):
        _hyperparameter = {name: value for name, value in zip(names, values)}

        # noise_threshold = 0.0015 * _hyperparameter["n_noise_topics"]

        configure_path = os.path.join(root_directory, "{0}.conf".format(index))
        output_path = os.path.join(root_directory, "{0}.npz".format(index))
        name = "grid_search_parameter_LDA_{0}".format(index)
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
