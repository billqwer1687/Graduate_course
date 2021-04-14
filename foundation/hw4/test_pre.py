import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
def main():
    f = open('hw4_test.dat')
    array = []
    for line in f:
        array.append([float(x) for x in line.split()])
    f.close()
    array = np.asarray(array)
    y_label = array[:,-1]
    array_pre = np.zeros((300,21))
    k=0
    for i in range(6):
        for j in range(i,6):
            array_pre[:,k] = array[:,i]*array[:,j]
            k += 1
    array = np.delete(array,6,1)
    array_pre = np.c_[array,array_pre]
    array_pre = np.c_[np.ones(300),array_pre]
    array_pre = np.c_[y_label,array_pre]
    
    fd = open("test_data.dat","w")    
    string = ""
    for i in range(300):
        string += (str(array_pre[i][0]) + " ")
        for j in range(1,29):
            string += (str(j)+':'+str(round(array_pre[i][j],6))+" ")
        string += "\n"
    fd.write(string)
if __name__ == "__main__":
    main()