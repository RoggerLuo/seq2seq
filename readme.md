哪还有时间研究算法，  
时间都用来读说明书了  

### tf.train.LoggingTensorHook
Inherits From: SessionRunHook  

``` python
__init__(
    tensors,
    every_n_iter=None,
    every_n_secs=None,
    at_end=False,
    formatter=None
)
```

	
Args:

- tensors: 一个list，里面是tensor的名字.
- every_n_iter: 很明显了.
- every_n_secs: iter和secs必须选一个.
- at_end: bool specifying whether to print the values of tensors at the end of the run.
- formatter: function, 传入{name:tensor}的dict，返回一个字符串. 

SessionRunHook还有那些实例？我也不知道  


