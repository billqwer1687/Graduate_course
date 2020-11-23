import cv2
import numpy as np
def h1(c,d):
    if c == d:
        return c
    else:
        return 'b'
def h2(a,i):
    if a== i:
        return 1
    return 0
def h3(b,c,d,e):
    if b == c and (d == b and e == b):
        return 'r'
    elif b == c and (d != b or e != b):
        return 'q'
    else:
        return 's'
    
def Mark_Interior_Border(d_img):
    mib_img = np.zeros((64,64))
    for i in range(64):
        for j in range(64):
            if d_img[i][j] > 0:
                x1,x2,x3,x4 = 0,0,0,0
                if i == 0: 
                    if j == 0:
                        x1,x4 = d_img[i][j+1] , d_img[i+1][j]
                    elif j == 63:
                        x3,x4 = d_img[i][j-1] , d_img[i+1][j]
                    else:
                        x1,x3,x4 = d_img[i][j+1],d_img[i][j-1] , d_img[i+1][j]
                elif i == 63:
                    if j == 0:
                        x1,x2 = d_img[i][j+1] , d_img[i-1][j]
                    elif j == 63:
                        x2,x3 = d_img[i-1][j] , d_img[i][j-1]
                    else:
                        x1,x2,x3 = d_img[i][j+1],d_img[i-1][j],d_img[i][j-1]
                else:
                    if j == 0:
                        x1,x2,x4 = d_img[i][j+1],d_img[i-1][j],[i+1][j]
                    elif j ==63:
                        x2,x3,x4 = d_img[i-1][j],d_img[i][j-1],d_img[i+1][j]
                    else:
                        x1,x2,x3,x4 = d_img[i][j+1],d_img[i-1][j],d_img[i][j-1],d_img[i+1][j]
                a1 = h1(1, x1)
                a2 = h1(a1, x2)
                a3 = h1(a2, x3)
                a4 = h1(a3, x4)
                
                if a4 == 'b':
                    mib_img[i][j] = 2
                else:
                    mib_img[i][j] = 1
    return mib_img
def Pair_Relation_Operation(mib_img):
    pair_img = np.zeros((64,64))
    for i in range(64):
        for j in range(64):
            if mib_img[i][j] > 0:
                x1,x2,x3,x4 = 0,0,0,0
                if i == 0: 
                    if j == 0:
                        x1,x4 = mib_img[i][j+1] , mib_img[i+1][j]
                    elif j == 63:
                        x3,x4 = mib_img[i][j-1] ,mib_img[i+1][j]
                    else:
                        x1,x3,x4 = mib_img[i][j+1],mib_img[i][j-1] , mib_img[i+1][j]
                elif i == 63:
                    if j == 0:
                        x1,x2 = mib_img[i][j+1] , mib_img[i-1][j]
                    elif j == 63:
                        x2,x3 = mib_img[i-1][j] , mib_img[i][j-1]
                    else:
                        x1,x2,x3 = mib_img[i][j+1],mib_img[i-1][j],mib_img[i][j-1]
                else:
                    if j == 0:
                        x1,x2,x4 = mib_img[i][j+1],mib_img[i-1][j],mib_img[i+1][j]
                    elif j ==63:
                        x2,x3,x4 = mib_img[i-1][j],mib_img[i][j-1],mib_img[i+1][j]
                    else:
                        x1,x2,x3,x4 = mib_img[i][j+1],mib_img[i-1][j],mib_img[i][j-1],mib_img[i+1][j]              
                if h2(x1,1) + h2(x2,1) + h2(x3,1) + h2(x4,1) >= 1 and mib_img[i][j] == 2:
                    pair_img[i][j] = 1
                else:
                    pair_img[i][j] = 2
    return pair_img
def Yokoi_Number(d_img):
    Yokoi_img = np.zeros((64,64))
    for i in range(64):
        for j in range(64):
            if d_img[i][j] > 0:
                if i == 0:
                    if j == 0:
                        x7,x2,x6 = 0,0,0
                        x3,x0,x1 = 0,d_img[i][j],d_img[i][j+1]
                        x8,x4,x5 = 0,d_img[i+1][j],d_img[i+1][j+1]
                    elif j == 63:
                        x7,x2,x6 = 0,0,0
                        x3,x0,x1 = d_img[i][j-1],d_img[i][j],0
                        x8,x4,x5 = d_img[i+1][j-1],d_img[i+1][j],0
                    else:
                        x7,x2,x6 = 0,0,0
                        x3,x0,x1 = d_img[i][j-1],d_img[i][j],d_img[i][j+1]
                        x8,x4,x5 = d_img[i+1][j-1],d_img[i+1][j],d_img[i+1][j+1]
                elif i == 63:
                    if j == 0:
                        x7,x2,x6 = 0,d_img[i-1][j],d_img[i-1][j+1]
                        x3,x0,x1 = 0,d_img[i][j],d_img[i][j+1]
                        x8,x4,x5 = 0,0,0
                    elif j == 63:
                        x7,x2,x6 = d_img[i-1][j-1],d_img[i-1][j],0
                        x3,x0,x1 = d_img[i][j-1],d_img[i][j],0
                        x8,x4,x5 = 0,0,0
                    else:
                        x7,x2,x6 = d_img[i-1][j-1],d_img[i-1][j],d_img[i-1][j+1]
                        x3,x0,x1 = d_img[i][j-1],d_img[i][j],d_img[i][j+1]
                        x8,x4,x5 = 0,0,0
                else:
                    if j == 0:
                        x7,x2,x6 = 0,d_img[i-1][j],d_img[i-1][j+1]
                        x3,x0,x1 = 0,d_img[i][j]  ,d_img[i][j+1] 
                        x8,x4,x5 = 0,d_img[i+1][j],d_img[i+1][j+1]
                    elif j == 63:
                        x7,x2,x6 = d_img[i-1][j-1],d_img[i-1][j],0
                        x3,x0,x1 = d_img[i][j-1]  ,d_img[i][j]  ,0
                        x8,x4,x5 = d_img[i+1][j-1],d_img[i+1][j],0
                    else:
                        x7,x2,x6 = d_img[i-1][j-1],d_img[i-1][j],d_img[i-1][j+1]
                        x3,x0,x1 = d_img[i][j-1]  ,d_img[i][j]  ,d_img[i][j+1]
                        x8,x4,x5 = d_img[i+1][j-1],d_img[i+1][j],d_img[i+1][j+1]
                a1 = h3(x0,x1,x6,x2)
                a2 = h3(x0,x2,x7,x3)
                a3 = h3(x0,x3,x8,x4)
                a4 = h3(x0,x4,x5,x1)
                if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
                    Yokoi_img[i][j] = 5
                else:
                    result = 0
                    for element in [a1,a2,a3,a4]:
                        if element == 'q':
                            result = result + 1
                    Yokoi_img[i][j] = result
    return Yokoi_img
                
                


def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
    img_bin = np.zeros((512,512))
    for i in range(512):
        for j in range(512):
            if img[i][j] < 128:
                img_bin[i][j] = 0
            else:
                img_bin[i][j] = 1
    d_img = np.zeros((64,64),np.int)
    for i in range(64):
        for j in range(64):
            d_img[i][j] = img_bin[i*8][j*8]
    thin = np.zeros((64,64))
    thin = d_img
    change = 1
    while change:
        change = 0
        mib_img = Mark_Interior_Border(thin)
        pair_img = Pair_Relation_Operation(mib_img)
        Yokoi_img = Yokoi_Number(d_img)
        for i in range(64):
            for j in range(64):
                if Yokoi_img[i][j] ==1 and pair_img[i][j] == 1:
                    thin[i][j] = 0
                    change = 1
                    
    for i in range(64):
        for j in range(64):
            if thin[i][j] == 1:
                thin[i][j] = 255
    cv2.imwrite('lena.thinned.bmp', thin)
                

    
if __name__ == "__main__":
    main()