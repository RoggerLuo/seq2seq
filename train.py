from seq2seq import seq2seq
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

import timeline

from logging_hook import get_logging_hook
import training_data as t_data
import config
import pkl
import logging # 如果不加logging

tf.logging._logger.setLevel(logging.INFO) # 和tf.logging，logging_hook信息就打印不出来

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

input_max_length,output_max_length = config.get_max()
model_dir='model/seq2seq'

def load_vocab(filename):
    pklData = pkl.read(filename) # list ['我','是']
    vocab = {}
    for idx, item in enumerate(pklData):
        vocab[item.strip()] = idx
    return vocab # 一个对象{key,value} value是idx

def predict(est):
    """
    因为把output当成feature传入，而不是labels
    所以numpy_input_fn中y不填

    predict的时候output用不到，但是必须填,
    因为用predict和train用的同一个模型，
    
    如果没有output，执行模型的时候会报错,
    output的shape不对，也会报错

    报错很奇怪
    Length of tensors in x and y is mismatched

    """
    input_ = np.array([[1,2],[3,4]])
    output_ = np.array([[100,2],[3,4]]) # 要符合seq2seq模型中tf.concat的要求，shape必须是(batch_size,x_length)
    x={'input':input_,'output':output_}
    
    """
    把所有的数据变成一个巨大的batch传进入x，它会自己划分成更小的batch
    假设原作者手动拼出来的batch数据,size为2,（即t_data.input_fn这个函数所生成的batch数据）是 
        [
            item1,
            item2
        ]
    那么传入的格式是 
        [
            item,
            item,
            item,
            item,
            ...
        ]
    如果设置的batch_size为3
    那么，feature['input']将会得到
        [item,item,item]
    每次执行next，都会拿取一个新的batch，如果list中的数据全部用完了，那就会重新从头部开始拿取
    """
    input_fn = tf.estimator.inputs.numpy_input_fn(x, y=None, batch_size=1, shuffle=False, num_epochs=10)
    predictions = est.predict(input_fn=input_fn)
    print(predictions)
    print(next(predictions))
    print(next(predictions))
    # 再往后，没有数据了，batch_size就只能循环以前的数据，
    print(next(predictions))
    print(next(predictions))
    print(next(predictions))

def train(est):
    # Make hooks to print examples of inputs/predictions.
    print_inputs = get_logging_hook(['input_0', 'output_0'],vocab) 
    print_predictions = get_logging_hook(['predictions', 'train_pred'],vocab) # predictions和train_pred是啥 不一样吗
    timeline_hook = timeline.TimelineHook(model_dir, every_n_iter=100)
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


if __name__ == "__main__":
    vocab = load_vocab('./vocab.pkl') 
    params = {
        'vocab_size': len(vocab),
        'embed_dim': 100, # embed_dim和num_units可以不同？
        'num_units': 256
    }    
    # 'batch_size': 32,
    est = tf.estimator.Estimator(
        model_fn=seq2seq,
        model_dir=model_dir, 
        params=params
    )
    train(est)
    # predict(est)


