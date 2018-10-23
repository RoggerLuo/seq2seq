### 好的经验
看完理论就去github、google上找tensorflow的实现代码  
边跑起来边学  理解更透彻，  

python debugger工具还没有

把tensorflow、numpy和python的api、知识 分分类，总结起来

一个阶段结束，
继续下一个阶段的学习

###  还有很多问题 

训练的时候 是整个段落丢进去  
还是按某个长度切断 ？


attention机制还不熟悉  

estimator也是一言难尽  
各个api只是搭在一起跑起来了  
并不知道每个api和它的参数的详细用法

最后，词向量 输入和输出用同一个矩阵的，  
输入和输出用一样的句子，会有什么效果？  

如果想做一个问答机器人  
要怎么训练

摘要机器人呢？

如果想要它自己学习资料呢？ 

---

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


