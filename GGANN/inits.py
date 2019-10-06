import numpy as np
from numpy import ndarray
from typing import  List

# 极小数，将a/b转化成a/(b+SMALL_NUMBER),防止b为0
SMALL_NUMBER = 1e-7

# 权重初始化方法
def glorot_init(shape) ->ndarray:
    """
    Glorot & Bengio (AISTATS 2010) init.
    :param shape: [n,m] 权重维度
    :return:
    """
    initialization_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)




