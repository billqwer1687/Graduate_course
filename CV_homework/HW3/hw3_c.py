import cv2
import numpy 
from PIL import Image  
import matplotlib.pyplot as plt
im = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)

array = [0]*256
for i in range(512):
    for j in range(512):
        im[i][j] = im[i][j]/3
        array[im[i][j]] = array[im[i][j]] + 1
s = [0]*256
for i in range(256):
    temp = 0
    for j in range(i):
        temp = temp + array[j]/(512*512)
    s[i] = 255*temp
for i in range(512):
    for j in range(512):
        im[i][j] = s[im[i][j]]
cv2.imwrite('hw3_c.jpg',im)
hgm = [0]*256
for x in range(512):
    for y in range(512):
        hgm[im[x,y]] = hgm[im[x,y]] + 1
plt.bar(numpy.arange(len(hgm)),hgm,width=1.0)
plt.savefig('hw3_c_hist.jpg')
