# -*- coding: utf-8 -*-
import itertools
import os
import pickle
import shutil
import numpy as np
from sklearn.cross_validation import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from mdl.topic_model.HDP.blei_hdp_c_util import write_data_to_file
from mdl.topic_model.ArtificialDataGenerator import split_matrix_by_col


_run_qsub_script = \
    """for textfile in $( ls . | grep .sub$ ); do qsub "${{textfile}}"; done;
    """
_hdp_script = \
    """
    for i in {{1..{n_trials}}}
    do
    mkdir {root_directory}/$i
    {HDP_PATH}  --train_data {root_directory}/train.dat --directory {root_directory}/$i --save_lag -1 --max_iter 2000 > /dev/null
    {HDP_PATH} --test_data {root_directory}/test.dat --model_prefix {root_directory}/$i/final --directory {root_directory}/$i --save_lag -1  --max_iter 1000 > /dev/null
    done
    """
#{taqy_index:02d}
_qsub_script = """
#!/bin/bash
#PBS -l nodes=1:ppn={n_jobs}:taqy
#PBS -o {root_directory}/log_{index}.out
#PBS -e {root_directory}/log_{index}.err
#PBS -N {name}

export MKL_NUM_THREADS={n_jobs}
export OMP_NUM_THREADS={n_jobs}

python ~/Dropbox/programming/research/pycharm/MDL/grid_search_parameter_LDA_real_data.py -conf {configure_path}
"""
# {taqy_index:02d}

HDP_PATH = ""
_mdl_path = os.path.join(os.path.dirname(__file__), "../../dataset/pre_calculated_multinomial_mdl.pkl")
_result_path = os.path.join(os.path.dirname(__file__), "../../temp")
_doc_path = os.path.join(os.path.dirname(__file__), "../../dataset")

_row_data_name = "data_row"
_row_test_name = "test_row"
_col_data_name = "data_col"
_test_data_name = "data_test"

if __name__ == '__main__':
    for cluster in range(2, 5):
    # for cluster in range(2, 7):
        _raw_data_name = 'Reuters/r{0}_stemmed_success'.format(cluster)
        # _raw_data_name = 'reuters_{0}.npz'.format(cluster)

        n_trials = 10
        _random_state = None
        _test_size = 0.01
        _topics = range(1, 11, 1)
        # _topics = range(1, 13, 1)
        _raw_data_path = os.path.join(_doc_path, _raw_data_name)

        n_jobs = 1
        n_sample_trials = 3
        mdl_path = _mdl_path
        root_directory = os.path.join(_result_path, "LDA_multi_old_" + os.path.splitext(os.path.basename(_raw_data_name))[0])
        _row_data_path = os.path.join(root_directory, _row_data_name)
        _row_test_path = os.path.join(root_directory, _row_test_name)
        _col_data_path = os.path.join(root_directory, _col_data_name)
        _test_data_paht = os.path.join(root_directory, _test_data_name)
        #
        if not os.path.exists(root_directory):
            os.makedirs(root_directory)
            # raise ValueError
        else:
            for _file in [os.path.join(root_directory, name) for name in os.listdir(root_directory)]:
                if os.path.isfile(_file):
                    # if _file.split(".")[-1] in ["sub", "conf"]:
                    os.remove(_file)
                else:
                    shutil.rmtree(_file)

        with open(_raw_data_path, "rb") as f:
            _X, _y, _words = pickle.load(f)

        # split data by row
        if _y is None:
            _X_train, _X_test = train_test_split(_X, test_size=_test_size, random_state=_random_state)
        else:
            _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=_test_size, random_state=_random_state)
        with open(_row_data_path, "wb") as f:
            pickle.dump((_X_train, _X_test), f)
        if _y is not None:
            with open(_row_test_path, "wb") as f:
                pickle.dump((_y_train, _y_test), f)
        write_data_to_file(_X_train, _X_test, root_directory)

        index = 0
        _topic_index = 0
        _step = 1
        while _topic_index < len(_topics):
            para_map = {"n_topics": [_topics[_topic_index+i] for i in range(min(_step, len(_topics)-_topic_index))]}
            _topic_index += min(_step, len(_topics)-_topic_index)
            data_path = _row_data_path
            configure_path = os.path.join(root_directory, "{0}.conf".format(index))
            output_path = os.path.join(root_directory, "{0}.npz".format(index))
            name = _raw_data_name + "_{0}".format(index)
            _base = {name: value for name, value in locals().items() if
                     name[0] != "_" and type(value) in [float, int, str, dict, list]}

            with open(configure_path, "wb") as f:
                pickle.dump(_base, f)

            # _taqy_index = (np.random.randint(1, 10000) % 18) + 1
            # while _taqy_index in [3, 4, 5, 10]:
            #     _taqy_index = np.random.randint(1, 10000) % 18 + 1

            with open(os.path.join(root_directory, "{0}.sub".format(index)), "w") as f:
                # _body = _qsub_script.format(taqy_index=_taqy_index, **_base)
                _body = _qsub_script.format(**_base)
                # taqy_index=_taqy_index,
                f.write(_body)

            index += 1
            print(_base)

        with open(os.path.join(root_directory, "{0}.sub".format(index)), "w") as f:
            _body = _hdp_script.format(**_base)
            f.write(_body)

        # # split data by col
        # _X_train, _X_test = split_matrix_by_col(_X, test_size=_test_size)
        # with open(_col_data_path, "wb") as f:
        #     pickle.dump((_X_train, None), f)
        # with open(_test_data_paht, "wb") as f:
        #     pickle.dump(_X_test, f)
        #
        # index = 0
        # _topic_index = 0
        # while _topic_index < len(_topics):
        #     para_map = {"n_topics": [_topics[_topic_index+i] for i in range(min(_step, len(_topics)-_topic_index))]}
        #     _topic_index += min(_step, len(_topics)-_topic_index)
        #     data_path = _col_data_path
        #     configure_path = os.path.join(root_directory, "{0}.col.conf".format(index))
        #     output_path = os.path.join(root_directory, "{0}.col.npz".format(index))
        #     name = "LDA_real_data_col_{0}".format(index)
        #     _base = {name: value for name, value in locals().items() if
        #              name[0] != "_" and type(value) in [float, int, str, dict, list]}
        #
        #     with open(configure_path, "wb") as f:
        #         pickle.dump(_base, f)
        #
        #     with open(os.path.join(root_directory, "{0}.col.sub".format(index)), "w") as f:
        #         _body = _qsub_script.format(**_base)
        #         f.write(_body)
        #
        #     index += 1
        #     print(_base)

        with open(os.path.join(root_directory, "run_qsub.sh"), "w") as f:
            f.write(_run_qsub_script.format(**_base))
