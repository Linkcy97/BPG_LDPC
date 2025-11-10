# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2025-11-10 17:49:07
# LastEditors  : Chongyang Li
# LastEditTime : 2025-11-10 17:58:32
# FilePath     : /BPG_LDPC/test.py

import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducability
sionna.phy.config.seed = 42
import matplotlib.pyplot as plt
import numpy as np
from sionna.phy import Block
from sionna.phy.utils import ebnodb2no, compute_ser, compute_ber, PlotBER
from sionna.phy.channel import FlatFadingChannel, KroneckerModel
from sionna.phy.channel.utils import exp_corr_mat
from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.mapping import SymbolDemapper, Mapper, Demapper, BinarySource, QAMSource
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder

num_tx_ant = 4
num_rx_ant = 16
num_bits_per_symbol = 4
batch_size = 1024
qam_source = QAMSource(num_bits_per_symbol)
x = qam_source([batch_size, num_tx_ant])
print(x.shape)

channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=True)
no = 0.2 # Noise variance of the channel
no = sionna.phy.utils.ebnodb2no(5, 2**num_bits_per_symbol, 0.5)  # Example conversion from EbNo to noise variance

# y and h are the channel output and channel realizations, respectively.
y, h = channel(x, no)
print(y.shape)
print(h.shape)
