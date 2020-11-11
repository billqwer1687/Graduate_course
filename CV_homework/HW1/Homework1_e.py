import cv2

im = cv2.imread("lena.bmp")

half_img = cv2.resize(im,(0,0),fx=0.5,fy=0.5)

cv2.imwrite("shrink_half.bmp",half_img)
