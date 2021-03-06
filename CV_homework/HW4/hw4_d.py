import cv2
import numpy as np
def erosion(im,kernel):
    im_01 = np.zeros((512,512),dtype=np.int)   
    for i in range(512):
        for j in range(512):
            if im[i][j] < 128:
                im[i][j] = 0
                im_01[i][j] = 0
            else:
                im[i][j] = 255
                im_01[i][j] = 1
    for i in range(512):
        for j in range(512):
            if im_01[i][j] == 1:
                for item in kernel:
                    p,q = item
                    if i+p <512 and j+q <512:
                        if im_01[i+p][j+q] == 0:
                            im[i][j] = 0
def dilation(im,kernel):
    im_01 = np.zeros((512,512),dtype=np.int)   
    for i in range(512):
        for j in range(512):
            if im[i][j] < 128:
                im[i][j] = 0
                im_01[i][j] = 0
            else:
                im[i][j] = 255
                im_01[i][j] = 1
    for i in range(512):
        for j in range(512):
            if im_01[i][j] == 1:
                for item in kernel:
                    p,q = item
                    if i+p <512 and j+q <512:
                        im[i+p][j+q] = 255
def main():
    im = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]
    dilation(im,kernel)
    erosion(im,kernel)
    
    cv2.imwrite('closing.bmp', im)

if __name__ == "__main__":
    main()