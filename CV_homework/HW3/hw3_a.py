import cv2
from PIL import Image  
import numpy 
import matplotlib.pyplot as plt

im = Image.open('lena.bmp')
im_array = numpy.array(im)
hgm = [0]*256

for x in range(512):
    for y in range(512):
        hgm[im_array[x,y]] = hgm[im_array[x,y]] + 1
plt.bar(numpy.arange(len(hgm)),hgm,width=1.0)
plt.savefig('histogram.jpg')