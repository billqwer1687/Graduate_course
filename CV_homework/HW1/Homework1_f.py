import cv2

img = cv2.imread("lena.bmp")
ret,binary_im = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

cv2.imwrite("binary_lean.bmp",binary_im)
