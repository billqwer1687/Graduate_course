import cv2
import rawpy
import numpy as np
def Padding(img):
    padding_img = np.zeros((img.shape[0]+2,img.shape[1]+2))
    padding_img[1:-1, 1:-1] = img[:, :] 
    padding_img[0, 1:-1] = img[0, :]    
    padding_img[-1, 1:-1] = img[-1, :]  
    padding_img[1:-1, 0] = img[:, 0]    
    padding_img[1:-1, -1] = img[:, -1]  
    padding_img[0, 0] = img[0, 0]       
    padding_img[0, -1] = img[0, -1]     
    padding_img[-1, 0] = img[-1, 0]     
    padding_img[-1, -1] = img[-1, -1]  
    return padding_img
def progress(cur, total):
    print('%.2f%%'%(100 * cur / total), end='\r')

def Compute_demosaicking(Pad_img,number_image):
	Demo_img = np.zeros((number_image,Pad_img.shape[1],Pad_img.shape[2],3))
	
	for k in range(1):
		print(k)
		for j in range(1,Pad_img.shape[2]-1):
			for i in range(1,Pad_img.shape[1]-1):

				progress(j * Pad_img.shape[1] + i, Pad_img.shape[1] * Pad_img.shape[2])
				if(i%2 == 1 and (i+j)%2==0): #if red

					Demo_img[k][i][j][0] = (Pad_img[k][i-1][j-1] + Pad_img[k][i-1][j+1] + Pad_img[k][i+1][j-1] + Pad_img[k][i+1][j+1])/4
					Demo_img[k][i][j][1] = (Pad_img[k][i][j-1] + Pad_img[k][i-1][j] + Pad_img[k][i][j+1] + Pad_img[k][i+1][j])/4
					Demo_img[k][i][j][2] = Pad_img[k][i][j]
				elif((i+j)%2==1): #if green
					if(i%2 == 1):

						Demo_img[k][i][j][0] = (Pad_img[k][i-1][j] + Pad_img[k][i+1][j] )/2
						Demo_img[k][i][j][1] = Pad_img[k][i][j]
						Demo_img[k][i][j][2] = (Pad_img[k][i][j-1] + Pad_img[k][i][j+1] )/2
					else:
						Demo_img[k][i][j][0] = (Pad_img[k][i][j-1] + Pad_img[k][i][j+1] )/2
						Demo_img[k][i][j][1] = Pad_img[k][i][j]
						Demo_img[k][i][j][2] = (Pad_img[k][i-1][j] + Pad_img[k][i+1][j] )/2

				elif(i%2 == 0 and (i+j)%2==0): #if blue
					Demo_img[k][i][j][0] = Pad_img[k][i][j]
					Demo_img[k][i][j][1] = (Pad_img[k][i][j-1] + Pad_img[k][i-1][j] + Pad_img[k][i][j+1] + Pad_img[k][i+1][j])/4
					Demo_img[k][i][j][2] = (Pad_img[k][i-1][j-1] + Pad_img[k][i-1][j+1] + Pad_img[k][i+1][j-1] + Pad_img[k][i+1][j+1])/4

	return Demo_img[:,1:Demo_img.shape[1],1:Demo_img.shape[2],:]


def Compute_radiance(img,B):
	radiance_img = np.zeros((img.shape[1],img.shape[2]),dtype=np.float32)
	for i in range(img.shape[1]):
		for j in range(img.shape[2]):
			w = 0
			progress(i * img.shape[2] + j, img.shape[1] * img.shape[2])
			for k in range(img.shape[0]):
				w += (img[k][i][j]*B[k])
			radiance_img[i][j] = (w)
	return radiance_img




def main():
	#load data
	number_image = 1
	img = np.zeros((10,3464,5202))
	raw_img = (rawpy.imread("VFX/"+str(10)+".cr2"))
	raw_img = raw_img.raw_image_visible
	img[0] = raw_img
	#shutter speed
	B = [0.0167]

	Pad_img = np.zeros((number_image,img.shape[1]+2,img.shape[2]+2))
	for i in range(number_image):
		Pad_img[i] = Padding(img[i])
	img_b = np.zeros((number_image,img.shape[1],img.shape[2]))
	img_g = np.zeros((number_image,img.shape[1],img.shape[2]))
	img_r = np.zeros((number_image,img.shape[1],img.shape[2]))
	print(Pad_img.shape)
	#Demosaik_img = Pad_img
	Demosaik_img = Compute_demosaicking(Pad_img,number_image)


	

	com_img_b = np.zeros((number_image,1024,1536))
	com_img_g = np.zeros((number_image,1024,1536))
	com_img_r = np.zeros((number_image,1024,1536))

	for i in range(number_image):
		com_img_b[i] = cv2.resize(Demosaik_img[i,:,:,0], (1536, 1024), interpolation=cv2.INTER_CUBIC)
		com_img_g[i] = cv2.resize(Demosaik_img[i,:,:,1], (1536, 1024), interpolation=cv2.INTER_CUBIC)
		com_img_r[i] = cv2.resize(Demosaik_img[i,:,:,2], (1536, 1024), interpolation=cv2.INTER_CUBIC)
	
	radiance_b = Compute_radiance(com_img_b,B)
	radiance_g = Compute_radiance(com_img_g,B)
	radiance_r = Compute_radiance(com_img_r,B)

	radiance = cv2.merge([radiance_b, radiance_g, radiance_r])
	print(radiance)
	cv2.imwrite('raw.hdr',radiance)

	radiance = radiance.astype("float32")
	tonemap = cv2.createTonemapReinhard(gamma=1.25)
	ldr = tonemap.process(radiance)
	ldr = np.clip(ldr*255, 0, 255).astype('uint8')


	cv2.imwrite('raw_ldr.jpg',ldr)
	





	
	



if __name__ == '__main__':
	main()