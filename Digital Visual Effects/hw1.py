import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import argparse

number_image = 10
number_pixel = 100

def weight(PixelValue):
	p_min, p_max = 0., 255.
	if PixelValue <= 127:
		return PixelValue - p_min + 1
	return p_max - PixelValue + 1


def np_weight(img):
	np_w = np.where(img <= 127, 255 - img + 1, img - 0 + 1)
	return np_w


def random_pixel(img,number_image,number):
	Z = np.zeros((number,number_image))
	for n in range(number):
		i = random.randint(10,img.shape[1]-10)
		j = random.randint(10,img.shape[2]-10)

		while is_not_flat(img[5], i , j, 5, 100) or is_not_ascending(img, i, j):
			i = random.randint(10,img.shape[1]-10)
			j = random.randint(10,img.shape[2]-10)
		for k in range(number_image):
			Z[n][k] = img[k][i][j]
	
	Z = Z.astype(int)
	return Z


def is_not_flat(img, i, j, k_size, thres):
    neighbors = img[i - k_size // 2: i + k_size // 2 + 1, j - k_size // 2: j + k_size // 2 + 1]
    diff = sum(sum(abs(neighbors - np.full_like(neighbors, img[i, j]))))
    if diff <= thres:
        return False
    else:
        return True


def is_not_ascending(imgs, i, j):
    for k in range(number_image - 1):
        if imgs[k][i][j] > imgs[k + 1][i][j]:
            return True
    return False


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
'''	
def Compute_radiance(img,g,B):
	radiance_img = np.zeros((img.shape[1],img.shape[2]),dtype=np.float32)
	print(img.shape)
	for i in range(img.shape[1]):
		for j in range(img.shape[2]):
			w = 0
			w_sum = 0
			for k in range(img.shape[0]):
				w += weight(img[k][i][j]) * (g[img[k][i][j]] - B[k])
				w_sum += weight(img[k][i][j])
			radiance_img[i][j] = (w/w_sum)
	radiance_img = np.exp(radiance_img)
	return radiance_img
'''
	
def Compute_radiance(img,g,B):
	radiance_img = np.zeros((img.shape[1],img.shape[2]),dtype=np.float32)
	print(img.shape)
	r_sum = np.zeros((img.shape[1],img.shape[2]))
	w_sum = np.zeros((img.shape[1],img.shape[2]))
	for k in range(img.shape[0]):
		im_1d = img[k,:,:].flatten()

		r = (g[im_1d] - B[k]).reshape(img.shape[1],img.shape[2])
		w = np_weight(im_1d)
		w = w.reshape(img.shape[1],img.shape[2])
		w_r = r * w
		r_sum += w_r
		w_sum += w
	radiance_img = np.exp(r_sum/w_sum)
	return radiance_img


def gaussian_blurs(im, smax=25, a=0.5, fi=8.0, epsilon=0.01):
	cols, rows = im.shape
	blur_prev = im
	num_s = int((smax+1)/2)
	blur_list = np.zeros(im.shape + (num_s,))
	Vs_list = np.zeros(im.shape + (num_s,))
	for i, s in enumerate(range(1, smax+1, 2)):
		blur = cv2.GaussianBlur(im, (s, s), 0)
		Vs = np.abs((blur - blur_prev) / (2 ** fi * a / s ** 2 + blur_prev))
		blur_list[:, :, i] = blur
		Vs_list[:, :, i] = Vs
		# 2D index
		smax = np.argmax(Vs_list > epsilon, axis=2)
		smax[np.where(smax == 0)] = num_s
		smax -= 1
		# select blur size for each pixel
		I, J = np.ogrid[:cols, :rows]
		blur_smax = blur_list[I, J, smax]
		return blur_smax


def photographic_local(hdr):
	ldr = np.zeros_like(hdr, dtype=np.float32)
	Lw_ave = np.exp(np.mean(np.log(0.00000001 + hdr)))
	for c in range(3):
		Lw = hdr[:, :, c]
		Lm = (0.1 / Lw_ave) * Lw
		Ls = gaussian_blurs(Lm)
		Ld = Lm / (1 + Ls)
		ldr[:, :, c] = np.clip(np.array(Ld * 255), 0, 255)
	return ldr.astype(np.uint8)


def fit_E(Z, G, st):
	Wz = np_weight(Z).reshape(number_image, -1)
	Gz = G[Z].reshape(number_image,-1)

	upper = np.sum(Wz * Gz * st, axis=0).astype(np.float32)
	bottom = np.sum(Wz * st * st, axis=0).astype(np.float32) + 1e-8
	return upper / bottom


def fit_G(Z, G, E, st):
	Z = Z.reshape(number_image, -1)
	Wz = np_weight(Z).reshape(number_image, -1)
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
def robertson_method(Z_bgr, B, initG, epochs):
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




def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--tone_algo", type=int,
						help="0: using opencv Reinhard algorithm\n"
						 	"1: usiing self-made Reinhard algorithn(global)\n"
							"2: using self-made Reinhard algorithm(local)")
	parser.add_argument("--algo", type=int,
						help="0: using Debevec algorithm for radiance image\n"
							"1: using Robertson algorithm for radiance image")

	args = parser.parse_args()
	tone_mode = args.tone_algo
	algo = args.algo
	#load data
	img = []
	img_b = []
	img_g = []
	img_r = []
	
	for i in range(1,11):
		img.append(cv2.imread("source_images/"+str(i)+".jpg"))
	img = np.array(img)
	alignMTB = cv2.createAlignMTB()
	alignMTB.process(img, img)
	
	for i in range(number_image):
		b, g, r = cv2.split(img[i])
		img_b.append(b.astype(np.int32))
		img_g.append(g.astype(np.int32))
		img_r.append(r.astype(np.int32))
	img_b = np.array(img_b)
	img_g = np.array(img_g)
	img_r = np.array(img_r)
	
	if algo == 0:
		Z_b = random_pixel(img_b,number_image,number_pixel)
		Z_g = random_pixel(img_g,number_image,number_pixel)
		Z_r = random_pixel(img_r,number_image,number_pixel)

		B = [0.00025, 0.0004, 0.000625, 0.001, 0.0015625, 0.0025, 0.004, 0.00625, 0.01, 0.0167]
		B = np.log(B)

		g_b = Compute_lsq(Z_b,B)
		g_g = Compute_lsq(Z_g,B)
		g_r = Compute_lsq(Z_r,B)
		#'''
		plt.plot(range(256),g_b)
		plt.plot(range(256),g_g)
		plt.plot(range(256),g_r)
		plt.savefig("Debevec_response_curve.jpg")
		#'''
		radiance_b = Compute_radiance(img_b, g_b, B)
		radiance_g = Compute_radiance(img_g, g_g, B)
		radiance_r = Compute_radiance(img_r, g_r, B)

		radiance = cv2.merge([radiance_b, radiance_g, radiance_r])
		cv2.imwrite('Debevec.hdr',radiance)

		if tone_mode == 0:
			radiance = radiance.astype("float32")
			tonemap = cv2.createTonemapReinhard(gamma=1.25)
			ldr = tonemap.process(radiance)
			ldr = np.clip(ldr*255, 0, 255).astype('uint8')
			cv2.imwrite('Debevec_cv_ldr.jpg',ldr)

		elif tone_mode == 1:
			key = 0.3
			l_w = radiance
			l_bar = np.exp(np.mean(np.log(0.00000001+l_w)))
			l_m = (key/l_bar) * radiance
			l_w = np.max(l_m)
			l_d = (l_m*(1+l_m/(l_w**2))/(1+l_m))
			ldr = np.clip(np.array(l_d*255),0,255)
			ldr = ldr.astype(np.uint8)
			cv2.imwrite("Debevec_Reinhard_global.jpg",ldr)

		elif tone_mode == 2:
			ldr_local = photographic_local(radiance)
			cv2.imwrite("Debevec_Reinhard_local.jpg",ldr_local)

	elif algo == 1:
		B = np.array([0.00025, 0.0004, 0.000625, 0.001, 0.0015625, 0.0025, 0.004, 0.00625, 0.01, 0.0167])
		G_b_robertson = robertson_method(img_b, B, np.array([np.arange(0, 1, 1 / 256)]).reshape(256,1) , epochs=2)
		G_g_robertson = robertson_method(img_g, B, np.array([np.arange(0, 1, 1 / 256)]).reshape(256,1) , epochs=2)
		G_r_robertson = robertson_method(img_r, B, np.array([np.arange(0, 1, 1 / 256)]).reshape(256,1) , epochs=2)

		plt.plot(range(256),G_b_robertson)
		plt.plot(range(256),G_g_robertson)
		plt.plot(range(256),G_r_robertson)
		plt.savefig("Robertson_response_curve.jpg")


		radiance_b = Compute_radiance(img_b, G_b_robertson, B)
		radiance_g = Compute_radiance(img_g, G_g_robertson, B)
		radiance_r = Compute_radiance(img_r, G_r_robertson, B)

		radiance = cv2.merge([radiance_b, radiance_g, radiance_r])
		cv2.imwrite('Robertson.hdr',radiance)
	
		if tone_mode == 0:
			radiance = radiance.astype("float32")
			tonemap = cv2.createTonemapReinhard(gamma=1.25)
			ldr = tonemap.process(radiance)
			ldr = np.clip(ldr*255, 0, 255).astype('uint8')
			cv2.imwrite('Robertson_cv_ldr.jpg',ldr)
		elif tone_mode == 1:
			key = 0.4
			l_w = radiance
			l_bar = np.exp(np.mean(np.log(0.00000001+l_w)))
			l_m = (key/l_bar) * radiance
			l_w = np.max(l_m)
			l_d = (l_m*(1+l_m/(l_w**2))/(1+l_m))
			ldr = np.clip(np.array(l_d*255),0,255)
			ldr = ldr.astype(np.uint8)
			cv2.imwrite("Robertson_Reinhard_global.jpg",ldr)
		elif tone_mode == 2:
			ldr_local = photographic_local(radiance)
			cv2.imwrite("Robertson_Reinhard_local.jpg",ldr_local)




if __name__ == '__main__':
	main()