from seq2seq import seq2seq
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

import timeline

from logging_hook import get_logging_hook
import training_data as t_data
import config

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

input_max_length,output_max_length = config.get_max()

def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab # 一个对象{key,value} value是idx
vocab = load_vocab('vocab') 


def train_seq2seq(model_dir):
    # 一个对象{key,value} value是idx
    # 有start end unknown3个token
    params = {
        'vocab_size': len(vocab),
        'batch_size': 32,
        'embed_dim': 100, # embed_dim和num_units可以不同？
        'num_units': 256
    }    
    
    # Make hooks to print examples of inputs/predictions.
    print_inputs = get_logging_hook(['input_0', 'output_0'],vocab) 
    print_predictions = get_logging_hook(['predictions', 'train_pred'],vocab) # predictions和train_pred是啥 不一样吗
    timeline_hook = timeline.TimelineHook(model_dir, every_n_iter=100)

    est = tf.estimator.Estimator(
        model_fn=seq2seq,
        model_dir=model_dir, 
        params=params
    )

    est.train(
        input_fn=t_data.input_fn,
        hooks=[ # 4个hook
            tf.train.FeedFnHook(t_data.get_feed_fn(vocab)), 
            print_inputs, 
            print_predictions,
            timeline_hook
        ], 
        steps=10000
    )

import logging # 如果不加logging
tf.logging._logger.setLevel(logging.INFO) # 和tf.logging，logging_hook信息就打印不出来
train_seq2seq(model_dir='model/seq2seq')


# if __name__ == "__main__":
#     main()

