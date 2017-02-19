# -*- coding: utf-8 -*-

import argparse
import os

from .blei_hdp_c_util import get_log_likelihood_from_file, get_topic_number_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is to calculate score')
    parser.add_argument("-i", "--input", type=str, required=True, help="input directory")
    args = parser.parse_args()

    input_dir = args.input
    train_log_path = os.path.join(input_dir, "state.log")
    n_topics = get_topic_number_from_file(train_log_path)

    test_log_path = os.path.join(input_dir, "test.log")
    get_log_likelihood_from_file(test_log_path)
    n_

