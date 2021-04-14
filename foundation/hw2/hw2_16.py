import numpy as np
import random
def gen_data():
    array = np.zeros((2,2))
    for i in range(2):
        array[i][0] = random.uniform(-1,1)
        array[i][1] = np.sign(array[i][0])
    s = [-1,+1]
    min_value = 1
    for i in range(array.shape[0]-1):
        theta = [-1 , (array[i][0]+array[i+1][0])/2]
        for p in range(2):
            for q in range(2):
                hit = 0
                for j in range(array.shape[0]):
                    result = s[p] * np.sign(array[j][0]-theta[q])
                    if result == array[j][1]:
                        hit = hit +1
                if 1-(hit/array.shape[0]) < min_value:
                    min_value = 1-(hit/array.shape[0])
                    s_final = s[p]
                    t_final = theta[q]
    return s_final,t_final
    
def main():
    test_data = np.zeros((1000,2))
    total_Ein_Eout = 0
    total_theta = 0
    for i in range(1000):
        test_data[i][0] = random.uniform(-1,1)
        test_data[i][1] = np.sign(test_data[i][0])
    for i in range(10000):
        s,t = gen_data()
        total_theta = total_theta + abs(t/2)
        
    print(total_theta/10000)
    
if __name__ == '__main__':
    main()
