# -*- coding: utf-8 -*-

import os
import subprocess
import time
import re

import numpy as np
import pandas

MAX_TRY_NUMBER = 20
TRAIN_DATA_FILE_NAME = "train.dat"
TEST_DATA_FILE_NAME = "test.dat"


def matrix_to_lda_c(X):
    """matrix to lda-c format"""
    documents = [[] for _ in range(X.shape[0])]
    for d, w in zip(*X.nonzero()):
        documents[d].append("{0}:{1}".format(w, int(X[d, w])))
    lines = []
    for d in documents:
        lines.append("{0} {1}\n".format(len(d), " ".join(d)))
    return lines


def lines_to_file(lines, file_path):
    with open(file_path, "w") as f:
        f.writelines(lines)


def write_data_to_file(X_train, X_test, output_directory, prefix=""):
    if X_train is not None:
        lines_to_file(matrix_to_lda_c(X_train), os.path.join(output_directory, prefix + TRAIN_DATA_FILE_NAME))
    if X_test is not None:
        lines_to_file(matrix_to_lda_c(X_test), os.path.join(output_directory, prefix + TEST_DATA_FILE_NAME))


def get_topic_number_from_file(result_path):
    df = pandas.read_csv(result_path, encoding='utf-8', delim_whitespace=True)
    start_point = int(df.shape[0] / 4 * 3)
    return df["num.topics"][start_point:].value_counts().idxmax()


def get_log_likelihood_from_file(result_path):
    df = pandas.read_csv(result_path, encoding='utf-8', delim_whitespace=True)
    start_point = int(df.shape[0] / 4 * 3)
    return np.exp(-np.mean(df["avg.likelihood"][start_point:]))
    # return np.exp(-np.mean(df["avg.likelihood"][:1:1]))

#
# def get_log_likelihood_from_string(string):
#     likelihood = list(map(float, re.findall(r'-[0-9]+\.[0-9]+', string)))
#     return np.mean(likelihood[-201::50])
#
#
# def fit_and_score(X_train, X_test, temp_file_path):
#     hdp_path = "./hdp/hdp/hdp"
#
#     train_data_path = os.path.join(temp_file_path, TRAIN_DATA_FILE_NAME)
#     test_data_path = os.path.join(temp_file_path, TEST_DATA_FILE_NAME)
#     lines_to_file(matrix_to_lda_c(X_train), train_data_path)
#     lines_to_file(matrix_to_lda_c(X_test), test_data_path)
#
#     trained_model_path = os.path.join(temp_file_path, "mode.bin")
#     train_log_path = os.path.join(temp_file_path, "state.log")
#     test_log_path = os.path.join(temp_file_path, "test.log")
#
#     try_count = 0
#     while 1:
#         cmd = "{0} --algorithm train  --save_lag -1 --max_iter 2000 --data {1} --directory {2} --saved_model {3}  > /dev/null".format(
#             hdp_path, train_data_path, temp_file_path, trained_model_path)
#         p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout_data, stderr_data = p.communicate()
#         if p.returncode != 0:
#             try_count += 1
#             print(stderr_data.decode('utf-8'))
#             if try_count == MAX_TRY_NUMBER:
#                 # raise ValueError(stderr_data.decode('utf-8'))
#                 return -1, 0
#         else:
#             break
#
#     time.sleep(1)
#     n_topics = get_topic_number_from_file(train_log_path)
#
#     try_count = 0
#     while 1:
#         cmd = "{0} --algorithm test  --save_lag -1 --data {1} --directory {2} --saved_model {3}  > /dev/null".format(
#             hdp_path, test_data_path, temp_file_path, trained_model_path)
#
#         p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout_data, stderr_data = p.communicate()
#         if p.returncode != 0:
#             try_count += 1
#             print(stderr_data.decode('utf-8'))
#             if try_count == MAX_TRY_NUMBER:
#                 return -1, 0
#                 # raise ValueError(stderr_data.decode('utf-8'))
#         else:
#             break
#
#     # stdout_string = stdout_data.decode('utf-8')
#     # log_likelihood = get_log_likelihood_from_string(stdout_string)
#     log_likelihood = get_log_likelihood_from_file(test_log_path)
#     perplexity = np.exp(-log_likelihood / np.sum(X_test))
#     return n_topics, perplexity
