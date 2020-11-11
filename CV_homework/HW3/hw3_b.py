import cv2
import numpy 
from PIL import Image  
import matplotlib.pyplot as plt
im = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)

for i in range(512):
    for j in range(512):
        im[i][j] = im[i][j]/3

cv2.imwrite('hw3_b.jpg',im)
hgm = [0]*256

for x in range(512):
    for y in range(512):
        hgm[im[x,y]] = hgm[im[x,y]] + 1
plt.bar(numpy.arange(len(hgm)),hgm,width=1.0)
plt.savefig('hw3_b_hist.jpg')