import numpy as np
import tensorflow as tf
import config
GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

input_max_length,output_max_length = config.get_max()
batch_size = config.get_size()
def input_fn():
    # 创建placeholder 然后切片放进打印机打印
    inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
    output = tf.placeholder(tf.int64, shape=[None, None], name='output') # output 指label吗
    tf.identity(inp[0], 'input_0') # 好像内存中多了一个叫做input_0的op，然后到时候就可以根据名字调用
    tf.identity(output[0], 'output_0')
    return {'input': inp,'output': output}, None # 这个None是怎么回事

def get_feed_fn(vocab):
    input_filename='input'
    output_filename='output'
    # 每次输入的样本，长一点好还是短一点好呢？
        # 会不会长一点不太好预测
        # 短一点的会不会就捕捉不到句子之间的语义关系
    # 一篇文章要一个词一个词的分割吗
        # 每隔两个词就当一成一个新的sample，一句话可以有10个sample # 文字生成器
        # 还是一篇文章只用一次?
    def str2idx(string):
        string = string.split(' ') # 为了适应demo, 之后注释这一句
        return [vocab.get(token, UNK_TOKEN) for token in string]

    def sampler():
        while True:
            with open(input_filename) as finput:
                with open(output_filename) as foutput:
                    for in_line in finput:
                        out_line = foutput.readline()
                        # 这里输入原始的输入和输出，整理成函数
                        # 从离原始数据最近端开始
                        # 接口的数据结构 为分割线
                        yield {
                            'input': str2idx(in_line)[:input_max_length - 1] + [END_TOKEN],
                            'output': str2idx(out_line)[:output_max_length - 1] + [END_TOKEN]
                        }
    sample_me = sampler() # 生成idx的输入，并且切分好
    # return sample_me
    """
    FeedFnHook只做了一件事
    继承SessionRunHook,然后实现一个方法：
    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs(fetches=None, feed_dict=self.feed_fn()) # 注意这里
    """
    # 每次执行都给input output placeholder喂学习数据，batch size的数据
    # batch size在这里拼装, 
    def feed_fn(): # 每次sess.run都会重新执行feed_fn()
        inputs, outputs = [], []
        input_length, output_length = 0, 0
        for i in range(batch_size):
            res = next(sample_me)
            inputs.append(res['input']) # 把单个的素材 推进list
            outputs.append(res['output'])
            input_length = max(input_length, len(inputs[-1])) # inputs[-1]，inputs的最后一个，就是刚才append上的那一个 
            output_length = max(output_length, len(outputs[-1])) # 取batch中的最长值
        # Pad me right with </S> token.
        for i in range(batch_size):
            # 长度不够的地方填充end_token
            inputs[i] += [END_TOKEN] * (input_length - len(inputs[i])) # [7] * 5 = [7,7,7,7,7]
            outputs[i] += [END_TOKEN] * (output_length - len(outputs[i])) # [7,7,7,7,7] + [1,2,3] = [7 7 7 7 7 1 2 3]
        return {'input:0': inputs,'output:0': outputs} # return dict of tensor (feed_dict)
    return feed_fn
