# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist


def norm(label):
    label_vec = []
    label_value = label  
    for i in range(10):
        if i == label_value:
            label_vec.append(1)
        else:
            label_vec.append(0)
    return np.array(label_vec).reshape(-1, 1)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)
