import numpy as np
import imageio
import cv2
from einops import rearrange

x = np.arange(3 * 4 * 4).reshape(3, 4, 4)
print(x)
y = rearrange(x, 'c h w -> h w c')
print(y)
z = rearrange(x, 'c (h1 h2) (w1 w2) -> c (h1 w1) h2 w2',h1=2, w1=2)
print(z)
# Generate dummy random image with float values

# freeimage lib only supports float32 not float64 arrays

# img = imageio.imread('test.hdr',format='HDR-FI')
# img = cv2.imread("JPG/IMG_8257.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/1.jpg",img)
# img = cv2.imread("JPG/IMG_8258.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/2.jpg",img)
# img = cv2.imread("JPG/IMG_8259.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/3.jpg",img)
# img = cv2.imread("JPG/IMG_8260.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/4.jpg",img)
# img = cv2.imread("JPG/IMG_8261.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/5.jpg",img)
# img = cv2.imread("JPG/IMG_8262.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/6.jpg",img)
# img = cv2.imread("JPG/IMG_8263.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/7.jpg",img)
# img = cv2.imread("JPG/IMG_8264.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/8.jpg",img)
# img = cv2.imread("JPG/IMG_8265.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/9.jpg",img)
# img = cv2.imread("JPG/IMG_8266.JPG")
# img = cv2.resize(img, (1536, 1024), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("JPG/10.jpg",img)
# import math 
# def gaussian_blurs(im, smax=25, a=0.5, fi=8.0, epsilon=0.01):
# 	cols, rows = im.shape
# 	blur_prev = im
# 	num_s = int((smax+1)/2)
# 	blur_list = np.zeros(im.shape + (num_s,))
# 	Vs_list = np.zeros(im.shape + (num_s,))
# 	for i, s in enumerate(range(1, smax+1, 2)):
# 		blur = cv2.GaussianBlur(im, (s, s), 0)
# 		Vs = np.abs((blur - blur_prev) / (2 ** fi * a / s ** 2 + blur_prev))
# 		blur_list[:, :, i] = blur
# 		Vs_list[:, :, i] = Vs
# 		# 2D index
# 		smax = np.argmax(Vs_list > epsilon, axis=2)
# 		smax[np.where(smax == 0)] = num_s
# 		smax -= 1
# 		# select blur size for each pixel
# 		I, J = np.ogrid[:cols, :rows]
# 		blur_smax = blur_list[I, J, smax]
# 		return blur_smax
# def photographic_local(hdr):
# 	ldr = np.zeros_like(hdr, dtype=np.float32)
# 	Lw_ave = np.exp(np.mean(np.log(0.00000001 + hdr)))
# 	for c in range(3):
# 		Lw = hdr[:, :, c]
# 		Lm = (0.1 / Lw_ave) * Lw
# 		Ls = gaussian_blurs(Lm)
# 		Ld = Lm / (1 + Ls)
# 		ldr[:, :, c] = np.clip(np.array(Ld * 255), 0, 255)
# 	return ldr.astype(np.uint8)

# def main():
# 	radiance = cv2.imread('test.hdr',-1)
# 	key = 0.2
# 	l_w = radiance
# 	l_bar = np.exp(np.mean(np.log(0.00000001+l_w)))
# 	l_m = (key/l_bar) * radiance
# 	l_w = np.max(l_m)
# 	l_d = (l_m*(1+l_m/(l_w**2))/(1+l_m))
# 	ldr = np.clip(np.array(l_d*255),0,255)
# 	ldr = ldr.astype(np.uint8)
# 	cv2.imwrite("Reinhard_global.jpg",ldr)
# 	ldr_local = photographic_local(radiance)
# 	cv2.imwrite("Reinhard_local.jpg",ldr_local)

# 	radiance = radiance.astype("float32")
# 	tonemap = cv2.createTonemapReinhard(gamma=1.25)
# 	cldr = tonemap.process(radiance)
# 	cldr = np.clip(cldr*255, 0, 255).astype('uint8')
# 	cv2.imwrite("cv2_Reinhard.jpg",cldr)
upper = 0
	bottom = 0
	E = np.zeros((number_pixel,1))
	for i in range(number_pixel):
		for j in range(number_image):
			zij = Z[i][j]
			upper += weight(zij)*G[zij]*st[j]
			bottom += weight(zij)*st[j]*st[j]
		E[i] = upper/bottom
	return E
# if __name__ == '__main__':
# 	main()
