import numpy as np
from random import randrange
f = open('hw1_train.dat.txt')
array = []
for line in f:
    array.append([float(x) for x in line.split()])
f.close()
array = np.asarray(array)
size = array.shape[0]
array = np.c_[np.ones(size)*10,array]
result = []
for i in range(1000):
    W = np.zeros(11)
    N = 0
    update_count = 0
    while(N < 5*size):
        i = randrange(size)
        X = array[i][:-1]
        Y = np.dot(W,X)
        if(np.sign(Y) != array[i][-1]):
            W = W + array[i][-1]*np.array(X)
            update_count += 1
            N = 0
        else:
            N = N + 1
    result = np.append(result,update_count)
print(sorted(result)[500])
