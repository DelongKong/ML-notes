# -*- coding: utf-8 -*-

import numpy as np

# rule激活器
class ReluActivator(object):
    def forward(self, Z):    # 前向计算，计算输出
        return np.maximum(0, Z)

    def backward(self, output):  # 后向计算，计算导数
        return 1 if output > 0 else 0

# IdentityActivator激活器.f(x)=x
class IdentityActivator(object):
    def forward(self, Z):   # 前向计算，计算输出
        return Z

    def backward(self, output):   # 后向计算，计算导数
        return 1

#Sigmoid激活器
class SigmoidActivator(object):
    def forward(self, Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def backward(self, output):
        # return output * (1 - output)
        return np.multiply(output, (1 - output))  # 对应元素相乘

# tanh激活器
class TanhActivator(object):
    def forward(self, Z):
        return 2.0 / (1.0 + np.exp(-2 * Z)) - 1.0

    def backward(self, output):
        return 1 - output * output

# softmax激活器
class SoftmaxActivator(object):
    def forward(self, Z):  # 前向计算，计算输出
        return np.maximum(0, Z)

    def backward(self, output):  # 后向计算，计算导数
        return 1 if output > 0 else 0