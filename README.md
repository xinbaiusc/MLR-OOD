# MLR-OOD
MLR-OOD is a a Markov chain based Likelihood Ratio method for Out-Of-Distribution (OOD) detection of genomic sequences. 

## Installation
```
git clone https://github.com/xinbaiusc/MLR-OOD
cd MLR-OOD
```

## Prerequisites
To use MLR-OOD, two software packages: [genomic_ood](https://github.com/google-research/google-research/tree/master/genomics_ood) and [Afann](https://github.com/GeniusTang/Afann) are required as prerequisites. Please download and install these two software packages under the directory of ./MLR-OOD. Please refer to the corresponding GitHub pages for downloading these two software packages.

## Usage
The MLR-OOD method consists of three steps: training generative models based on LSTM for sequences in each in-distribution (ID) class, evaluation of the training models and calculating the likelihoods, and calculation of the prediction scores and prediction accuracy. The train.py and eval.py scripts for the first two steps are modified from scripts in [genomic_ood](https://github.com/google-research/google-research/tree/master/genomics_ood).

We use a toy dataset which does not contain real biological meanings to walk you through the whole process.

### Step 1: training generative models based on LSTM for sequences in each in-distribution class
We provide a toy dataset in ./train_data to illustrate the training process. Suppose there are three in-distribution classes, we train a generative model based on LSTM for each class separately. The input data for each ID class is a fasta file containing all the training sequences. Please note that all the training sequences must have the same length and each sequence must be in a single line. For example, we train for class 0 using the following command:
```
python -m train \
--hidden_lstm_size=30 \
--val_freq=100 \
--num_steps=1000 \
--in_tr_fasta_data=./train_data/in_tr_class0.fa \
--in_tr_fasta_data_class=0 \
--in_tr_data_dir=./train_data/tfrecord_class0 \
--out_dir=./train_data/tfrecord_class2/output
```
Here `--in_tr_fasta_data` specifies the directory containing the input sequence data in fasta format. `--in_tr_fasta_data_class` specifies the ID class index. `--in_tr_data_dir` specifies the directory to store the tfrecord format data converted from the fasta data. `--out_dir` specifies output directory for the model. For the details of hyperparameter tuning related to the generative model, please refer to the MLR-OOD paper.

You need to manually create these input and output directories before running the train.py script. Then you can follow the same steps to train models for the remaining in-distribution classes. We recommend using GPU resources for training in order to save the computational time.

### Step 2: evaluation of the training models and calculating the likelihoods
We provide another toy dataset in ./test_data to illustrate the evaluation process. The input of this step is a fasta file containing all the testing sequences, regardless of ID or OOD, and all the testing sequences must have the same length which must be a multiple of the training sequence length. The output is a vector of log likelihoods for each testing sequence. Since we train for each ID model separately, we also calculate the likelihood for each ID class separately. For example, the command for calculating the likelihood for class 0 is as follows:
```
python3 -m eval \
--model_dir_frgd=./train_data/tfrecord_class0/output/generative_l250_bs100_lr0.0005_hr30_nrFalse_regl2_regw0.000000_fi-1_mt0.00/model \
--n_samples=200 \
--ckpt_step=900 \
--test_fasta_data=./test_data/test_data.fa \
--test_data_dir=./test_data \
--testing_seq_len=250 \
--training_seq_len=250 \
--out_test_likelihood_file=./test_data/lstm_ll_class0.txt
```
Here `--model_dir_frgd` specifies the model directory containing the checkpoints trained from step 1. `n_samples` specifies the number of testing sequences. `--ckpt_step` specifies the number of epoches you would like to use. `test_fasta_data` specifies the directory containing the testing sequence data in fasta format. `--testing_seq_len` specifies the directory to store the tfrecord format data converted from the fasta data. `--out_test_likelihood_file` specifies the file name of the output likelihood vector. 

Then you can follow the same steps for other ID training classes.

### Step 3: calculating of the prediction scores and prediction accuracy (optional)
The final step is to calculate the prediction scores of MLR-OOD for all the testing sequences, and calculate the AUROC and AUPRC if labels are known. There are a few input parameters:
#### Options
```
-h, --help            show this help message and exit
-i, --data            A full path to the input testing sequence file
-o, --output          The path to output the prediction results
-f, --lstm            A full path to the text file storing the path to the LSTM likelihood files of each ID training class output by the eval step
-l, --label           (optional) A full path to the binary true label file (1 as ID, 0 as OOD)
-L0, --trlen          The sequence length for all input training sequences
-L1, --telen          The sequence length for all input testing sequences
-x, --order           The maximum possible MC order (default 3)
-t, --thread          The number of threads (default 1)
```
Before this step, you should prepare a text file storing the full path of LSTM likelihoods for all ID classes calculated from step 2, and the path of the text file is used as an input parameter. You should create the output path if it does not exist.

#### Command
If you have the true labels for the testing sequences, use a text file to store the labels (1 as ID, 0 as OOD) paring the testing sequences and input to MLR-OOD. One example useage on the toy dataset is:
```
python3 mlr_ood.py -i './test_data/test_data.fa' -o './output' -f './test_data/test_likelihood_read.txt' -l './test_data/test_label.txt' -L0 250 -L1 250
```
This will output both a file containing the prediction scores (greater values correspond to higher likelihood of being ID) and a file containing the prediction accuracy ([AUROC, AUPRC]).

MLR-OOD can also handle the case where we do not have prior knowledge on the labels of the testing sequences. One example useage on the toy dataset is:
```
python3 mlr_ood.py -i './test_data/test_data.fa' -o './output' -f './test_data/test_likelihood_read.txt' -L0 250 -L1 250
```
This will output a file containing the prediction scores.

## Real metagenomic datasets (bacterial, viral, and plasmid) for benchmarking
We compose three different microbial data sets consisting of bacterial, viral, and plasmid sequences for comprehensively benchmarking OOD detection methods. The bacterial dataset except the Test2018 dataset is downloaded from [genomic_ood](https://github.com/google-research/google-research/tree/master/genomics_ood). The Test2018 bacterial, viral, and plasmid datasets are new. All the datasets contain different lengths of sequences for testing. 

The datasets can be downloaded from [Gooogle Drive](https://drive.google.com/drive/folders/1Kz0kQ_D1VWYqA-GDld783O7H8AzNuHkC?usp=sharing). All sequences are in tfrecord format.

## Copyright and License Information
Copyright 2021 University of Southern California

Authors: Xin Bai, Jie Ren, Fengzhu Sun

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

For bugs or usage inqury, contact Xin Bai at [xinbai@usc.edu](xinbai@usc.edu). For other general questions, contact Dr. Fengzhu Sun at [fsun@usc.edu](fsun@usc.edu).
