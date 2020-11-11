import math

f = open("Input.dat")
array = []

for line in f:
    array.append([float(x) for x in line.split()]) #transfer input to array
f.close()

message_number = int(array[0][0])
R_array = [0] * message_number
tau = array[1][0]
max_B = [0] * message_number
#calculate block time
for i in range(message_number):
    for j in range(i,message_number):
        if(max_B[i] < array[j+2][1]):
            max_B[i] = array[j+2][1]

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
            R_array[i] = "No"
    R_array[i] = round(Q + array[i+2][1],2)
    print(R_array[i])
