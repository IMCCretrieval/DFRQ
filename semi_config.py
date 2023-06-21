import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import pickle
import scipy.io
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

# Dataset path
# Source: 5,000, Target: 54,000
# Gallery: 54,000 Query: 1,000
data_dir = '/home/lipd/DFRQ/cifar10'


# For training
ImagNet_pretrained_path = '/home/lipd/DFRQ/models/ImageNet_pretrained'
model_save_path = '/home/lipd/DFRQ/models/'

# For evaluation
model_load_path = '/home/lipd/DFRQ/models/final_model/24_bits_dfrq_set1_420.ckpt'
cifar10_label_sim_path = '/home/lipd/DFRQ/cifar10/cifar10_Similarity.mat'


n_CLASSES = 10
image_size = 32
img_channels = 3
n_DB = 54000

'Hyperparameters for training'
# Training epochs, 1 epoch represents training all the source data once
total_epochs = 500
batch_size = 500

count_fft=3
self_count=count_fft+1
unlabel_batch_size=int(500/(self_count)) #125 * 4
# save model for every save_term-th epoch
# save_term = 60

# length of codeword
# len_code = 24

# # Number of codebooks
# n_book = 4

# # Number of codewords=(2^bn_word)
# bn_word = 8
# intn_word = pow(2, bn_word)

# # Number of bits for retrieval
# n_bits = n_book * bn_word

# Soft assignment input scaling factor
alpha = 20.0



# ##ranking factor
# eta=2

# lam1, 2: loss function balancing parameters
lam_1 = 0.1
lam_2 = 0.1