import numpy as np
import math
import sys
from random import randrange
#np.set_printoptions(threshold=sys.maxsize)
def main():
    f = open('hw3_train.dat')
    array = []
    for line in f:
        array.append([float(x) for x in line.split()])
    f.close()
    array = np.asarray(array)
    # add column 0 and delete y column
    X = np.c_[np.ones(array.shape[0]),array]
    X = np.delete(X,11,1)
    #declare Y
    Y = array[:,-1]
    Y = Y.reshape(Y.shape[0],1)
    #psudo inverse
    X_inv = np.linalg.pinv(X)
    W_lin = np.matmul(X_inv,Y)
    err_in_wlin = 0
    for i in range(X.shape[0]):
        y_pre = 0
        y_pre = (np.matmul(X[i],W_lin)).item()
        err_in_wlin += (y_pre - Y[i][0])**2
    err_in_wlin = err_in_wlin/Y.shape[0]
    print("Problem 14:",err_in_wlin)

    tau = 0.001
    itr = 1
    cnt = 0
    for i in range(itr):
        Wt = np.zeros((1,11))
        ein_wt = 10 
        while ein_wt > 1.01*err_in_wlin: 
            ein_wt = 0 
            number = randrange(X.shape[0])
            Wt = Wt + 2* tau *(Y[number][0]-np.matmul(X[number],np.transpose(Wt)))*X[number]
            for i in range(X.shape[0]):
                y_pre = 0
                y_pre = (np.matmul(X[i],np.transpose(Wt))).item()
                ein_wt += (y_pre - Y[i][0])**2
            ein_wt = ein_wt/Y.shape[0]
            cnt += 1
    print("Problem15:",cnt/itr)
    ##P16
    tau = 0.001
    itr = 1000
    total_e = 0
    for i in range(itr):
        Wt = np.zeros((1,11))
        ein_wt = 0
        for i in range(500): 
            number = randrange(X.shape[0])
            Wt = Wt + tau *(1/(1+math.exp(Y[number][0]*np.matmul(X[number],np.transpose(Wt)))))*(Y[number][0]*X[number])
        for i in range(X.shape[0]):
            y_pre = 0
            y_pre = (np.matmul(X[i],np.transpose(Wt))).item()
            ein_wt += np.log(1+math.exp(-1*Y[i][0]*y_pre))
        ein_wt = ein_wt/Y.shape[0]
        total_e += ein_wt

    print("Problem16:",total_e/itr)
    tau = 0.001
    itr = 1000
    total_e = 0
    for i in range(itr):
        Wt = np.transpose(W_lin)
        ein_wt = 0
        for i in range(500): 
            number = randrange(X.shape[0])
            Wt = Wt + tau *(1/(1+math.exp(Y[number][0]*np.matmul(X[number],np.transpose(Wt)))))*(Y[number][0]*X[number])
        for i in range(X.shape[0]):
            y_pre = 0
            y_pre = (np.matmul(X[i],np.transpose(Wt))).item()
            ein_wt += np.log(1+math.exp(-1*Y[i][0]*y_pre))
        ein_wt = ein_wt/Y.shape[0]
        total_e += ein_wt

    print("Problem17:",total_e/itr)
    err_in_wlin = 0
    err_out_wlin = 0
    f = open('hw3_test.dat')
    array_t = []
    for line in f:
        array_t.append([float(x) for x in line.split()])
    f.close()
    array_t = np.asarray(array_t)
    # add column 0 and delete y column
    X_t = np.c_[np.ones(array_t.shape[0]),array_t]
    X_t = np.delete(X_t,11,1)
    #declare Y
    Y_t = array_t[:,-1]
    Y_t = Y_t.reshape(Y_t.shape[0],1)
    for i in range(X.shape[0]):
        y_pre = 0
        y_pre = (np.matmul(X[i],W_lin)).item()
        if y_pre > 0:
            y_pre = 1
        else:
            y_pre = -1
        if y_pre != Y[i][0]:
            err_in_wlin += 1
    err_in_wlin = err_in_wlin/Y.shape[0]

    for i in range(X_t.shape[0]):
        yt_pre = 0
        yt_pre = (np.matmul(X_t[i],W_lin)).item()
        if yt_pre > 0:
            yt_pre = 1
        else:
            yt_pre = -1
        if yt_pre != Y_t[i][0]:
            err_out_wlin += 1
    err_out_wlin = err_out_wlin/Y_t.shape[0]
    print("Problem 18:",err_out_wlin - err_in_wlin)
    err_in_wlin = 0
    err_out_wlin = 0

    for i in range(2,4):
        for j in range(10):
            X = np.c_[X , (X[:,j+1]**i)]
    
    X_inv = np.linalg.pinv(X)
    W_lin = np.matmul(X_inv,Y)
    for i in range(X.shape[0]):
        y_pre = 0
        y_pre = (np.matmul(X[i],W_lin)).item()
        if y_pre > 0:
            y_pre = 1
        else:
            y_pre = -1
        if y_pre != Y[i][0]:
            err_in_wlin += 1
    err_in_wlin = err_in_wlin/Y.shape[0]

    for i in range(2,4):
        for j in range(10):
            X_t = np.c_[X_t , (X_t[:,j+1]**i)]
    for i in range(X_t.shape[0]):
        yt_pre = 0
        yt_pre = (np.matmul(X_t[i],W_lin)).item()
        if yt_pre > 0:
            yt_pre = 1
        else:
            yt_pre = -1
        if yt_pre != Y_t[i][0]:
            err_out_wlin += 1
    err_out_wlin = err_out_wlin/Y_t.shape[0]
    print("Problem 19:",err_out_wlin - err_in_wlin)
    


    err_in_wlin = 0
    err_out_wlin = 0

    for i in range(2,11):
        for j in range(10):
            X = np.c_[X , (X[:,j+1]**i)]
    
    X_inv = np.linalg.pinv(X)
    W_lin = np.matmul(X_inv,Y)
    for i in range(X.shape[0]):
        y_pre = 0
        y_pre = (np.matmul(X[i],W_lin)).item()
        if y_pre > 0:
            y_pre = 1
        else:
            y_pre = -1
        if y_pre != Y[i][0]:
            err_in_wlin += 1
    err_in_wlin = err_in_wlin/Y.shape[0]

    for i in range(2,11):
        for j in range(10):
            X_t = np.c_[X_t , (X_t[:,j+1]**i)]
    for i in range(X_t.shape[0]):
        yt_pre = 0
        yt_pre = (np.matmul(X_t[i],W_lin)).item()
        if yt_pre > 0:
            yt_pre = 1
        else:
            yt_pre = -1
        if yt_pre != Y_t[i][0]:
            err_out_wlin += 1
    err_out_wlin = err_out_wlin/Y_t.shape[0]
    print("Problem 20:",err_out_wlin - err_in_wlin)
if __name__ == '__main__':
    main()

