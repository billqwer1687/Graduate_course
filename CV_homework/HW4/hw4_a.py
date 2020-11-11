import cv2
import numpy as np
def dilation(im,im_01,kernel):
    for i in range(512):
        for j in range(512):
            if im_01[i][j] == 1:
                for item in kernel:
                    p,q = item
                    if i+p <512 and j+q <512:
                        im[i+p][j+q] = 255
def main():
    im = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
    im_01 = np.zeros((512,512),dtype=np.int)
    for i in range(512):
        for j in range(512):
            if im[i][j] < 128:
                im[i][j] = 0
                im_01[i][j] = 0
            else:
                im[i][j] = 255
                im_01[i][j] = 1
    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]
    dilation(im,im_01,kernel)
    cv2.imwrite('dilation.bmp', im)


if __name__ == "__main__":
    main()