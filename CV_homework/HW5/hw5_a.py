import cv2
import numpy as np
def dilation(im,kernel):
    im_dil = np.zeros((512,512),dtype=np.int)
    for i in range(512):
        for j in range(512):
            if im[i][j] > 0:
                max_value = 0
                for item in kernel:
                    p,q = item
                    if 0<= i+p <512 and 0<= j+q <512:
                        if im[i+p][j+q] > max_value:
                            max_value = im[i+p][j+q]
                for item in kernel:
                    p,q = item
                    if 0 <= i+p < 512 and 0 <= j+q < 512:
                        im_dil[i+p][j+q] = max_value
    return im_dil
def main():
    im_in = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]
    im_out = dilation(im_in,kernel)
    cv2.imwrite('dilation.bmp', im_out)


if __name__ == "__main__":
    main()