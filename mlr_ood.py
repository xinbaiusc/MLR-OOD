# Title             :mlr_ood.py
# Description       :Detecting out-of-distribution genomic sequences using a Markov chain based likelihood ratio method
# Author            :Xin Bai
# Contact           :xinbai@usc.edu
# Version           :1.0.0

import os, sys, optparse, argparse
import numpy as np
from math import log
from sklearn import metrics
from sklearn.metrics import average_precision_score
sys.path.append('./Afann')
from src._count import kmer_count
from src._count import kmer_count_seq
from src._count import kmer_count_m_k
from src._count import kmer_count_m_k_seq

def get_sequences(seqfile):
    sequence_list = []
    with open(seqfile) as f:
        for line in f.readlines():
            if not line.startswith('>'):
                sequence_list.append(line.strip())
    return sequence_list

def BIC(seq, M, Num_Threads, Reverse, P_dir, from_seq=False):
    M_count = kmer_count_seq(seq, M + 1, Num_Threads, False)
    S = []
    for i in range(M+1):
        M_count = M_count.reshape(4**M, 4)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            log = M_count * np.log(M_count/np.sum(M_count, axis=1)[:,np.newaxis])
            log[np.isnan(log)] = 0
            log = np.sum(log)
            bic = -2 * log + 3 * 4**M * np.log(np.sum(M_count))
        S = [bic] + S
        M_count = np.sum(M_count, axis=1)
        M -= 1
    return S.index(min(S))

def find_most_common_order(seqfile, max_possible_order = 3, num_threads = 1):
    sequence_list = get_sequences(seqfile)
    order_list = np.zeros(len(sequence_list))
    for idx, seq in enumerate(sequence_list):
        order_list[idx] = int(BIC(seq, max_possible_order, num_threads, Reverse = False, P_dir = ''))
    return np.argmax(np.bincount(order_list.astype(int)))

# This function computes the negative log MC likelihood according to the most common MC order divided by the sequence length. 
def compute_mc_neg_log_likelihood(seqfile, seq_len, max_possible_order = 3, num_threads = 1):
    sequence_list = get_sequences(seqfile)
    order = find_most_common_order(seqfile, max_possible_order, num_threads)
    likelihood_array = np.zeros(len(sequence_list))
    for seq_idx, seq in enumerate(sequence_list):
        if len(seq) != seq_len:
            raise ValueError('Each sequence must have the same length as the input length!')
        kmer_count = kmer_count_seq(seq, order + 1, num_threads, False)
        for kmer_idx, j in enumerate(kmer_count):
            if j != 0:
                div = kmer_idx // 4
                denominator_kmer = np.sum(kmer_count[div * 4 : (div + 1) * 4])
                likelihood_array[seq_idx] -= j*log(j / denominator_kmer, 4)/seq_len
    return likelihood_array

def read_lstm_from_file(filename):
    with open(filename) as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            if not os.path.exists(line):
                e = 'File %s do no exsits!'%line
                raise Exception(e)
            lstm_likelihood = np.loadtxt(line)
            if idx == 0:
                lstm_matrix = lstm_likelihood
            else:
                lstm_matrix = np.vstack((lstm_matrix, lstm_likelihood))
    return lstm_matrix

def average_lstm_testing_length(lstm_matrix, training_length, testing_length):
    ncol = len(lstm_matrix[0])
    len_ratio = int(testing_length/training_length)
    averaged_ncol = ncol // len_ratio
    if len_ratio != testing_length/training_length:
        raise ValueError('The testing sequence length must be a multiple of the training sequence length!')
    averaged_lstm_matrix = np.zeros([len(lstm_matrix), averaged_ncol])
    for i in range(len(lstm_matrix)):
        for j in range(averaged_ncol):
            averaged_lstm_matrix[i][j] = np.mean(lstm_matrix[i][(j * len_ratio):((j + 1) * len_ratio)])
    return averaged_lstm_matrix
      
def get_max_likelihood(lstm_matrix):
    return np.max(lstm_matrix, 0)

def write_pred_result_with_auc(max_lstm, mc_neg_log_likelihood, output_path, true_label_file):
    true_label = np.loadtxt(true_label_file)
    true_label = true_label.astype(int)
    if not len(max_lstm) == len(mc_neg_log_likelihood) == len(true_label):
        raise ValueError('The max LSTM likelihood, the MC likelihood, and the true label vectors must have the same length!')
    if not np.array_equal(true_label, true_label.astype(bool)):
        raise ValueError('The true label vector must be a binary (0 or 1) vector!')
    test_score = max_lstm + mc_neg_log_likelihood
    if not output_path.endswith('/'):
        output_path += '/'
    np.savetxt(output_path + "testscore.txt", test_score)
    auroc = metrics.roc_auc_score(true_label, test_score)
    auprc = average_precision_score(true_label, test_score)
    np.savetxt(output_path + "accuracy.txt", np.array([auroc, auprc]))

def write_pred_result(max_lstm, mc_neg_log_likelihood, output_path):
    if not len(max_lstm) == len(mc_neg_log_likelihood):
        raise ValueError('The max LSTM likelihood and the MC likelihood vectors must have the same length!')
    test_score = max_lstm + mc_neg_log_likelihood
    if not output_path.endswith('/'):
        output_path += '/'
    np.savetxt(output_path + "testscore.txt", test_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Example: python mlr_ood.py -i testFastaFile -o outputPath -m maxLstmFile -l trueLabelFile -L 250')
    parser.add_argument('-i', '--data', dest = "testing_fasta_sequence", help = "A full path to the input testing sequence file")
    parser.add_argument('-o', '--output', dest = "output_path", help = "The path to output the prediction results")
    parser.add_argument('-f', '--lstm', dest = "lstm_file", help = "A full path to the text file storing the path to the LSTM likelihood files of each ID training class output by the eval step")
    parser.add_argument('-l', '--label', dest = "true_label_file", help = "A full path to the binary true label file (1 as ID, 0 as OOD)")
    parser.add_argument('-L1', '--telen', dest = "testing_seq_len", help = "The sequence length for all input testing sequences")
    parser.add_argument('-L0', '--trlen', dest = "training_seq_len", help = "The sequence length for all training sequences")
    parser.add_argument('-x', '--order', dest = "maximum_possible_order", default = 3, help = "The maximum possible MC order")
    parser.add_argument('-t', '--thread', dest = "num_threads", default = 1, help = "The number of threads")
    args = parser.parse_args()
    seqfile = args.testing_fasta_sequence
    testing_seq_len = int(args.testing_seq_len)
    training_seq_len = int(args.training_seq_len)
    max_possible_order = args.maximum_possible_order
    num_threads = args.num_threads
    mc_neg_log_likelihood = compute_mc_neg_log_likelihood(seqfile, testing_seq_len, max_possible_order, num_threads)
    lstm_file = args.lstm_file
    lstm_matrix = read_lstm_from_file(lstm_file)
    averaged_lstm_matrix = average_lstm_testing_length(lstm_matrix, training_seq_len, testing_seq_len)
    max_lstm = get_max_likelihood(averaged_lstm_matrix)
    output_path = args.output_path
    if args.true_label_file:
        true_label_file = args.true_label_file
        write_pred_result_with_auc(max_lstm, mc_neg_log_likelihood, output_path, true_label_file)
    else:
        write_pred_result(max_lstm, mc_neg_log_likelihood, output_path)