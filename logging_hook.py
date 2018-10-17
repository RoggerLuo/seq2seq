import tensorflow as tf

def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}

# tensor的值是一个一维list
def get_formatter(name_list, vocab): # 工厂函数 name_list应该说是operation name list
    rev_vocab = get_rev_vocab(vocab)

    def to_str(sequence): # tensor的值是一个一维list
        tokens = [rev_vocab.get(x, "<UNK>") for x in sequence] # 如果没有x就UNK
        return ' '.join(tokens)

    def format(values): # dict of tag->tensor, dict就是js中的对象,tag是key，tensor是值
        res = []
        for name in name_list:
            res.append("%s = %s" % (name, to_str(values[name])))
        return '\n'.join(res)
    return format 

#formatter=get_formatter(name_list, vocab) 
#formatter不用加也可以正确的打印
#用了formatter就是string，不用formatter就是打印list，没差
def get_logging_hook(name_list,vocab,every_n_iter=100):
    return tf.train.LoggingTensorHook(
        name_list, 
        every_n_iter=every_n_iter
    )

