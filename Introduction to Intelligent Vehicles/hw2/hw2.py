import math
import random
from copy import deepcopy
def cost(array):
    message_number = int(array[0][0])
    R_array = [0] * message_number
    tau = array[1][0]
    max_B = [0] * message_number
    #calculate block time
    for i in range(message_number):
        for j in range(i,message_number):
            if(max_B[i] < array[j+2][1]):
                max_B[i] = array[j+2][1]
    s = 0
    for i in range(message_number):
        Q = max_B[i]
        temp = 0
        #first time calculate
        for j in range(i):
            temp = temp + math.ceil((Q+tau)/array[j+2][2])*array[j+2][1]
        RHS = max_B[i] + temp
        while(RHS != Q):
            Q = RHS
            temp = 0
            for j in range(i):
                temp = temp + math.ceil((Q+tau)/array[j+2][2])*array[j+2][1]
            RHS = max_B[i] + temp
            if(RHS + array[i+2][1] > array[i+2][2]):
                return 0
        R_array[i] = round(Q + array[i+2][1],2)
        s = s + R_array[i]
#        print(R_array[i])
    return s
def main():
    f = open("Input.dat")
    array = []
    array_p = []

    for line in f:
        array.append([float(x) for x in line.split()]) #transfer input to array
    f.close()
    T = 100
    r = 0.9995
    s = cost(array)
    s_f = s
    count =0
    while T>1:
        array_p = deepcopy(array)
        swap_num = []
        number = range(2,19)
        swap_num = random.sample(number,2)
        array_p[swap_num[0]],array_p[swap_num[1]] = array_p[swap_num[1]],array_p[swap_num[0]]
        s_p = cost(array_p)
        c = s_p - s
        if s_p != 0:
            if s_p < s_f:
                s_f = s_p
            if c <= 0:
                s = s_p
                array = deepcopy(array_p)
            if c > 0:
                a = math.exp(-c/T)
                if random.uniform(0,1) < a:
                    s = s_p
                    array = deepcopy(array_p)
        else:
            c = 1000
            a = math.exp(-c/T)
            if random.uniform(0,1) < a:
                s = s_p
                array = deepcopy(array_p)
        #count = count + 1
        T = r*T
    print("Total value:",s_f)
    #print(count)
    for i in range(17):
        for j in range(17):
            if array[j+2][0] == i:
                print(j)
if __name__ == '__main__':
    main()
