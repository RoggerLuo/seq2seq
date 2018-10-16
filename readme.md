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
	
	tensors: tensor的名字，可循环的dict that maps string-valued tags to tensors/tensor names, or iterable of tensors/tensor names.
	
	every_n_iter: int, print the values of tensors once every N local steps taken on the current worker.
	every_n_secs: int or float, print the values of tensors once every N seconds. Exactly one of every_n_iter and every_n_secs should be provided.
	at_end: bool specifying whether to print the values of tensors at the end of the run.
	formatter: function, takes dict of tag->Tensor and returns a string. If None uses default printing all tensors.

SessionRunHook还有那些实例？我也不知道  


