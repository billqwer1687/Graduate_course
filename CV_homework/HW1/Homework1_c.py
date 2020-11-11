import PIL.Image
import numpy
from numpy import asarray

im = PIL.Image.open("lena.bmp")
im_array = asarray(im)
reverse_im = numpy.zeros(shape=(512,512))

for x in range(512):
    for y in range(512):
        reverse_im[x,y] = im_array[y,x]

img = PIL.Image.fromarray(reverse_im)
if img.mode != 'RGB':
    img = img.convert('RGB')
img.save(" diagonally flip.bmp","bmp")