import cv2
from scipy import ndimage
im = cv2.imread("lena.bmp")

rotate_45 = ndimage.rotate(im,315)

cv2.imwrite("rotate_45.bmp",rotate_45)
