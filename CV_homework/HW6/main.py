import numpy as np
import cv2
def h(b,c,d,e):
    if b == c and (d == b and e == b):
        return 'r'
    elif b == c and (d != b or e != b):
        return 'q'
    else:
        return 's'
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
                a1 = h(x0,x1,x6,x2)
                a2 = h(x0,x2,x7,x3)
                a3 = h(x0,x3,x8,x4)
                a4 = h(x0,x4,x5,x1)
                if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
                    result = 5
                else:
                    result = 0
                    for element in [a1,a2,a3,a4]:
                        if element == 'q':
                            result = result + 1
                print(result,end = '')
            else:
                print(' ',end = '')
            if j == 63:
                print(' ')

if __name__ == "__main__":
    main()
