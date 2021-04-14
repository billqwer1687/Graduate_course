#!/usr/bin/env python
# coding: utf-8

# VFX-homework 1 High Dynamic Range
# 
# 

# In[359]:


import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

B = np.array([0.00025, 0.0004, 0.000625, 0.001, 0.0015625, 0.0025, 0.004, 0.00625, 0.01, 0.0167])
number_image = 10
number_pixel = 100


# basic import

# In[360]:


def NPweight(Z):
	np_w = np.concatenate((np.arange(1,129),np.arange(1,129))[::-1],axis=0)
	return np_w[Z].astype(np.float32)


# In[361]:


def weight(PixelValue):
	p_min, p_max = 0., 255.
	if PixelValue <= 127:
		return PixelValue - p_min + 1
	return p_max - PixelValue + 1


# In[362]:


def random_pixel(img,number_image,number):
	Z = np.zeros((number,number_image))
	r = random.sample(range(img.shape[1]), number)
	c = random.sample(range(img.shape[2]), number)
	for i in range(number):
		for j in range(number_image):
			Z[i][j] = img[j][r[i]][c[i]]
	Z = Z.astype(int)
	return Z


# In[363]:


def Compute_lsq(Z,B):
	A = np.zeros((Z.shape[0] * Z.shape[1] + 255, 256 + Z.shape[0]))
	b = np.zeros((A.shape[0], 1))
	k = 0
	lamda = 100
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			zij = Z[i][j]
			wij = weight(zij)
			A[k][zij] = wij
			A[k][256+i] = -wij
			b[k][0] = wij * B[j]
			k = k + 1
	A[k][127] = 1
	k = k + 1
	for i in range(1,255):
		A[k][i-1] = lamda * weight(i-1)
		A[k][i]   = (-2) * lamda * weight(i)
		A[k][i+1] = lamda * weight(i+1)
		k = k + 1

	x = np.linalg.lstsq(A,b,rcond=None)[0]
	return x[0:256,0]


# In[364]:


def Compute_radiance(img,g,B):
	radiance_img = np.zeros((img.shape[1],img.shape[2]),dtype=np.float32)
	print(img.shape)
	r_sum = np.zeros((img.shape[1],img.shape[2]))
	w_sum = np.zeros((img.shape[1],img.shape[2]))
	for k in range(img.shape[0]):
		im_1d = img[k,:,:].flatten()

		r = (g[im_1d] - B[k]).reshape(img.shape[1],img.shape[2])
		w = NPweight(im_1d)
		w = w.reshape(img.shape[1],img.shape[2])
		w_r = r * w
		r_sum += w_r
		w_sum += w
	radiance_img = np.exp(r_sum/w_sum)
	return radiance_img


# In[365]:


def fit_E(Z, G, st):
	Wz = NPweight(Z).reshape(number_image, -1)
	Gz = G[Z].reshape(number_image,-1)

	upper = np.sum(Wz * Gz * st, axis=0).astype(np.float32)
	bottom = np.sum(Wz * st * st, axis=0).astype(np.float32) + 1e-8
	return upper / bottom

def fit_G(Z, G, E, st):
	Z = Z.reshape(number_image, -1)
	Wz = NPweight(Z).reshape(number_image, -1)
	Wz_Em_st = Wz * (E * st)
    
	for m in range(256):
		index = np.where(Z == m)
		upper = np.sum(Wz_Em_st[index]).astype(np.float32)
		lower = np.sum(Wz[index]).astype(np.float32) + 1e-8
        
		if lower > 0:
			G[m] = upper / lower
            
	G /= G[127]
	return G

# initG not need to be log
def robertson_method(Z_bgr, initG, epochs=2):
	G_bgr = np.array(initG)
	st = B.reshape(number_image, 1)
    
	Z = np.array(Z_bgr)
	G = np.array(initG)
        
	for e in range(epochs):
		print('epoch = ',e)
		# Compute Ei (energy of each pixel)
		E = fit_E(Z, G, st)
		# Compute Gm
		G = fit_G(Z, G, E, st)

	G_bgr = G
    
	return G_bgr.astype(np.float32)


# In[366]:


def main():
	#load data
	number_image = 10
	number_pixel = 100
	img = []
	img_b = []
	img_g = []
	img_r = []
	for i in range(1,11):
		img.append(cv2.imread("VFX/"+str(i)+".jpg"))
	img = np.array(img)
	alignMTB = cv2.createAlignMTB()
	alignMTB.process(img, img)
	#shutter speed
	B = [0.00025, 0.0004, 0.000625, 0.001, 0.0015625, 0.0025, 0.004, 0.00625, 0.01, 0.0167]
	#B = np.log(B)
	for i in range(number_image):
		b, g, r = cv2.split(img[i])
		img_b.append(b.astype(np.int32))
		img_g.append(g.astype(np.int32))
		img_r.append(r.astype(np.int32))
	img_b = np.array(img_b)
	img_g = np.array(img_g)
	img_r = np.array(img_r)

	Z_b = random_pixel(img_b,number_image,number_pixel)
	Z_g = random_pixel(img_g,number_image,number_pixel)
	Z_r = random_pixel(img_r,number_image,number_pixel)
    
	G_b_robertson = robertson_method(img_b, np.array([np.arange(0, 1, 1 / 256)]).reshape(256,1) , epochs=2)
	G_g_robertson = robertson_method(img_g, np.array([np.arange(0, 1, 1 / 256)]).reshape(256,1) , epochs=2)
	G_r_robertson = robertson_method(img_r, np.array([np.arange(0, 1, 1 / 256)]).reshape(256,1) , epochs=2)

	#g_b = Compute_lsq(Z_b,B)
	#g_g = Compute_lsq(Z_g,B)
	#g_r = Compute_lsq(Z_r,B)
	#'''
	plt.plot(range(256),G_b_robertson)
	plt.plot(range(256),G_g_robertson)
	plt.plot(range(256),G_r_robertson)
	plt.savefig("robertson_response_curve.jpg")
	#'''
	radiance_b = Compute_radiance(img_b, G_b_robertson, B)
	radiance_g = Compute_radiance(img_g, G_g_robertson, B)
	radiance_r = Compute_radiance(img_r, G_r_robertson, B)

	radiance = cv2.merge([radiance_b, radiance_g, radiance_r])
	cv2.imwrite('robertson.hdr',radiance.astype(np.float32))

	radiance = radiance.astype("float32")
	tonemap = cv2.createTonemapReinhard(gamma=1.4)
	ldr = tonemap.process(radiance)
	ldr = np.clip(ldr*255, 0, 255).astype('uint8')


	cv2.imwrite('robertson_ldr.jpg',ldr)




if __name__ == '__main__':
	main()

