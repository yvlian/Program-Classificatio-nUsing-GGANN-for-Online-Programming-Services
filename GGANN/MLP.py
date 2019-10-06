import  tensorflow as tf
import  numpy as np


class MLP(object):
    """
    多层感知机实现：多层全连接神经网络
    """
    def __init__(self, in_size, out_size, hid_sizes, drop_keep_prob):
        """
        :param in_size:
        :param out_size:
        :param hid_size:
        :param droup_keep_prob:
        """
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = drop_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        """
        构建MLP权重weights和biases参数
        :return:
        """
        dims = [self.in_size]+self.hid_sizes+[self.out_size]  # [in_size, hid_size1, hid_size2,....,out_size]
        """
       zip:   打包为元组的列表
             a = [1,2,3]   b = [4,5,6]   zipped = zip(a,b)=[(1, 4), (2, 5), (3, 6)]
        """
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name="MLP_W_Layer%i" % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name="MLP_b_layer%i" % i)
                  for (i, s) in enumerate(weight_sizes)]
        network_params = \
            {
                "weights": weights,
                "biases": biases
            }
        return network_params

    def init_weights(self, shape):
        """
        初始化权重参数方式
        :param shape:
        :return:
        """
        return np.sqrt(6.0/(shape[-2]+shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32)-1)

    def __call__(self, inputs):
        """
        WX+B运算
        :param inputs:
        :return:
        """
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden

