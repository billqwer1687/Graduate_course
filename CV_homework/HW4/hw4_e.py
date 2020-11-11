import cv2
import numpy as np
def erosion(im,kernel):
    im_01 = np.zeros((512,512),dtype=np.int)   
    for i in range(512):
        for j in range(512):
            if im[i][j] < 128:
                im_01[i][j] = 0
            else:
                im_01[i][j] = 1

    for i in range(512):
        for j in range(512):
            t = True
            for item in kernel:
                p,q = item
                if 0<= i+p <512 and 0 <= j+q <512:
                    if im_01[i+p][j+q] == 0:
                        t = False
                        im[i][j] = 0
            if t:
                im[i][j] = 255
    

def main():
    im = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
    for i in range(512):
        for j in range(512):
            if im[i][j] < 128:
                im[i][j] = 0
            else:
                im[i][j] = 255
    im_c = -im +255
    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]
    K = [[-1, 0], [-1, 1], [0, 1]]
    J = [[0, -1], [0, 0], [1, 0]]
    erosion(im,J) 
    erosion(im_c,K)
    cv2.imshow('1',im_c)
    for i in range(512):
        for j in range(512):
            if(im[i][j] == im_c[i][j] == 255):
                im[i][j] = 255
            else:
                im[i][j] = 0
    cv2.imwrite('hit and miss.bmp', im)

if __name__ == "__main__":
    main()