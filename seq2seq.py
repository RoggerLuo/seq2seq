import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

import timeline

from logging_hook import get_logging_hook
GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab # 一个对象{key,value} value是idx

def seq2seq(mode, features, labels, params):
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    input_max_length = params['input_max_length']
    output_max_length = params['output_max_length']

    inp = features['input']
    output = features['output']
    batch_size = tf.shape(inp)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inp, 1)), 1)
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
    input_embed = layers.embed_sequence(
        inp, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    output_embed = layers.embed_sequence(
        train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')

    cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)

    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
    # train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
    #     output_embed, output_lengths, embeddings, 0.3
    # )
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)

    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
            cell = tf.contrib.rnn.GRUCell(num_units=num_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=num_units / 2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size))
                #initial_state=encoder_final_state)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=output_max_length
            )
            return outputs[0]
    train_outputs = decode(train_helper, 'decode')
    pred_outputs = decode(pred_helper, 'decode', reuse=True)

    tf.identity(train_outputs.sample_id[0], name='train_pred')
    weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
    loss = tf.contrib.seq2seq.sequence_loss(
        train_outputs.rnn_output, output, weights=weights)
    train_op = layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer=params.get('optimizer', 'Adam'),
        learning_rate=params.get('learning_rate', 0.001),
        summaries=['loss', 'learning_rate'])

    tf.identity(pred_outputs.sample_id[0], name='predictions')
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_outputs.sample_id,
        loss=loss,
        train_op=train_op
    )


def tokenize_and_map(line, vocab):
    return [vocab.get(token, UNK_TOKEN) for token in line.split(' ')]


def make_input_fn(
        batch_size, input_filename, output_filename, vocab,
        input_max_length, output_max_length,
        input_process=tokenize_and_map, output_process=tokenize_and_map):
    
    # 创建placeholder 然后切片放进打印机打印
    def input_fn():
        inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
        output = tf.placeholder(tf.int64, shape=[None, None], name='output') # output 指label吗
        tf.identity(inp[0], 'input_0') # 好像内存中多了一个叫做input_0的op，然后到时候就可以根据名字调用
        tf.identity(output[0], 'output_0')
        return {'input': inp,'output': output}, None # 这个None是怎么回事

    def sampler():
        while True:
            with open(input_filename) as finput:
                with open(output_filename) as foutput:
                    for in_line in finput:
                        out_line = foutput.readline()
                        yield {
                            'input': input_process(in_line, vocab)[:input_max_length - 1] + [END_TOKEN],
                            'output': output_process(out_line, vocab)[:output_max_length - 1] + [END_TOKEN]
                        }

    sample_me = sampler()
    
    """
    FeedFnHook只做了一件事
    继承SessionRunHook,然后实现一个方法：
    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs(fetches=None, feed_dict=self.feed_fn()) # 注意这里
    """
    def feed_fn(): # 每次sess.run都会重新执行feed_fn()
        inputs, outputs = [], []
        input_length, output_length = 0, 0
        for i in range(batch_size):
            rec = next(sample_me)
            inputs.append(rec['input'])
            outputs.append(rec['output'])
            input_length = max(input_length, len(inputs[-1])) # inputs[-1]，inputs的最后一个，就是刚才append上的那一个 
            output_length = max(output_length, len(outputs[-1])) # 取batch中的最长值
        # Pad me right with </S> token.
        for i in range(batch_size):
            inputs[i] += [END_TOKEN] * (input_length - len(inputs[i])) # 长度不够的地方填充end_token
            outputs[i] += [END_TOKEN] * (output_length - len(outputs[i]))
        return {'input:0': inputs,'output:0': outputs} # return dict of tensor (feed_dict)

    return input_fn, feed_fn


def train_seq2seq(input_filename, output_filename, vocab_filename,model_dir):
    vocab = load_vocab(vocab_filename) 
    # 一个对象{key,value} value是idx
    # 有start end unknown3个token
    params = {
        'vocab_size': len(vocab),
        'batch_size': 32,
        'input_max_length': 30,
        'output_max_length': 30,
        'embed_dim': 100, # embed_dim和num_units可以不同？
        'num_units': 256
    }

    input_fn, feed_fn = make_input_fn(
        params['batch_size'],
        input_filename,
        output_filename,
        vocab, params['input_max_length'], params['output_max_length'])

    # Make hooks to print examples of inputs/predictions.
    print_inputs = get_logging_hook(['input_0', 'output_0'],vocab) 
    print_predictions = get_logging_hook(['predictions', 'train_pred'],vocab) # predictions和train_pred是啥 不一样吗
    timeline_hook = timeline.TimelineHook(model_dir, every_n_iter=100)

    est = tf.estimator.Estimator(
        model_fn=seq2seq,
        model_dir=model_dir, 
        params=params)

    est.train(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed_fn), print_inputs, print_predictions,timeline_hook], # 4个hook
        steps=10000
    )


def main():
    tf.logging._logger.setLevel(logging.INFO)
    train_seq2seq('input', 'output', 'vocab', 'model/seq2seq')


if __name__ == "__main__":
    main()

