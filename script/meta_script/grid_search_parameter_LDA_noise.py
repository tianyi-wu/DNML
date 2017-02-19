# -*- coding: utf-8 -*-
import itertools
import os
import pickle
import shutil

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
        {HDP_PATH}  --train_data {root_directory}/{index}/$step/$trial/train.dat --directory {root_directory}/{index}/$step/$trial --save_lag -1 --max_iter 1500 > /dev/null
        {HDP_PATH} --test_data {root_directory}/{index}/$step/$trial/test.dat --model_prefix {root_directory}/{index}/$step/$trial/final --directory {root_directory}/{index}/$step/$trial --save_lag -1  --max_iter 500 > /dev/null
        find {root_directory}/{index}/$step/$trial/ -type f -not -name '*.log' -print0 | xargs -0 rm --
    done
done
"""
# --sample_hyper

HDP_PATH = "/home/wty36/Dropbox/programming/research/pycharm/MDL/hdp/hdp-faster/hdp"
_mdl_path = os.path.join(os.path.dirname(__file__), "../../dataset/pre_calculated_multinomial_mdl.pkl")
_result_path = os.path.join(os.path.dirname(__file__), "../../temp")

if __name__ == '__main__':
    K = 5
    n_sample_array = [300]
    steps = len(n_sample_array)

    # V = 300
    # alpha = 0.1
    # beta = 0.1
    # n_noise_topics = 0
    # noise_threshold = 0.0

    _topics = range(1, 10, 1)
    para_map = {"n_topics": _topics}

    if_run_hdp = 1

    n_jobs = 1
    n_trial = 5
    n_sample_trial = 1
    verbose = 0

    root_directory = os.path.join(_result_path, "grid_search_LDA_noise_doc_10")
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
        "V": [300, 500],
        "alpha": [0.05, 0.1, 0.2],
        "beta": [0.05, 0.1, 0.2],
        # "noise_ratio": [0.001, 0.005],
        "noise_ratio": list(range(0, 11, 1)),
        "n_noise_topics": [5],
        "D": [600]
    }

    _names = sorted(_hyperparameters.keys(), reverse=True)
    for values in list(itertools.product(*[_hyperparameters[name] for name in _names])):
        _hyperparameter = {name: value for name, value in zip(_names, values)}

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
