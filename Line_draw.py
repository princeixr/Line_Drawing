import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import scipy.signal 
from PIL import Image 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread("boy.jpg")									#reading the blurred image
gray = rgb2gray(img)    										#converting it to gray scale
print(gray.shape)
# plt.imsave("grayblurry.png", gray, cmap = plt.get_cmap('gray')) #saving the gray image
image = np.array(gray)
image = np.reshape(image, image.shape)

blurr_kernel = (1/25)*np.ones((5,5))
image = scipy.signal.convolve2d(image, blurr_kernel, 'same')
shape = gray.shape
n_img = shape[0]
m_img = shape[1]

# def tangent(image):
sobel_x = [[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0,-12,-6],[4,8,0,-8,-4],[1,2,0,-2,-1]]
sobel_y = [[1,4,6,4,1],[2,8,12,8,2],[0,0,0,0,0],[-2,-8,-12,-8,-2],[-1,-4,-6,-4,-1]]
# sobel_x = [[-3,0,3],[-10,0,10],[-3,0,3]]
# sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
sobel_x = np.array(sobel_x)
# sobel_x = np.divide(sobel_x, 8)
# sobel_y = [[-3,-10,-3],[0,0,0],[3,10,3]]
# sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]		
sobel_y = np.array(sobel_y)
# sobel_y = np.divide(sobel_y, 8)

grad_x = scipy.signal.convolve2d(image, sobel_x, 'same') 	# gradient along x dirn
grad_y = scipy.signal.convolve2d(image, sobel_y, 'same')
	
grad = np.sqrt(grad_x**2 + grad_y**2) 						# magnitude of grad			
# print(grad)
# grad = grad       #/np.amax(grad)
plt.imshow(grad_x)
plt.show()
plt.imshow(grad_y)
plt.show()
plt.imsave("gradient1.png", grad, cmap = plt.get_cmap('gray'))
# theta = np.arctan(np.true_divide(grad_y, grad_y))
# theta = np.nan_to_num(theta)
# tang_x = abs(grad)*np.cos(theta + np.pi/2)
# tang_y = abs(grad)*np.sin(theta + np.pi/2)	

tang_x = -grad_y											# taking tangent 
tang_y = grad_x	
tang = np.sqrt(tang_x**2 + tang_y**2)						# taking magnitude of tangent vector

n = 5
grad_p = np.pad(grad, ((n//2,n//2), (n//2,n//2)), 'constant')				#padded with zero
tang_x_p = np.pad(tang_x, ((n//2,n//2), (n//2,n//2)), 'constant')
tang_y_p = np.pad(tang_y, ((n//2,n//2), (n//2,n//2)), 'constant')

tang_mag = np.sqrt(tang_x_p**2 + tang_y_p**2)								#taking magnitude of padded tangent vector
tang_x_p = np.divide(tang_x_p, tang_mag)
tang_y_p = np.divide(tang_y_p, tang_mag)

tang_x_p = np.nan_to_num(tang_x_p)
tang_y_p = np.nan_to_num(tang_y_p)

def T_x(x,y):

	grad_diff = np.zeros((n,n))													#storing the difference in gradient
	w_d = np.zeros((n,n))														#direction weight matrix
	phi = np.ones((n,n))
	t_curr_x = np.zeros((n,n))													#current tangent vector's x component
	t_curr_y = np.zeros((n,n))													#   "       "             y component
	for i in range(n):
		for j in range(n):
			grad_diff[i][j] = grad_p[x - n//2 + i][y-n//2+j] - grad_p[x][y]
			t_dot = tang_x_p[x-n//2+i][y-n//2+j]*tang_x_p[x][y] + tang_y_p[x-n//2+i][y-n//2+j]*tang_y_p[x][y]
			
			if t_dot > 0:
				phi[i][j] = 1*phi[i][j]
			else:
				phi[i][j] = -1*phi[i][j]

			w_d[i][j] = abs(t_dot)
			# print(w_d)
			t_curr_x[i][j] = tang_x_p[x-n//2+i][y-n//2+j]
			t_curr_y[i][j] = tang_y_p[x-n//2+i][y-n//2+j]

	t_curr_x = np.nan_to_num(t_curr_x)
	t_curr_y = np.nan_to_num(t_curr_y)
	eta = 1
	w_m = (1/2)*(1 + eta*np.tanh(grad_diff))
	w_s = np.ones((n,n))
	# w_s[0,0] = 0
	# w_s[0,-1] = 0
	# w_s[-1,0] = 0
	# w_s[-1,-1] = 0

	t_new_x = np.sum(phi * t_curr_x * w_s * w_m * w_d)
	t_new_y = np.sum(phi * t_curr_y * w_s * w_m * w_d)

	#weight_func = np.sum(w_d) 

	return t_new_x, t_new_y

tang_new_x = np.zeros((n_img,m_img))
tang_new_y = np.zeros((n_img,m_img))
# for n in range(3):
# for k in range(2):

# for i in range(n_img):
# 	for j in range(m_img):
# 		tang_new_x[i][j], tang_new_y[i][j] = T_x(i,j, tang_x, tang_y)
# plt.imshow(tang_new_y)
# plt.show()
# plt.imsave("tang_new"+str(k)+".png", tang_new, cmap = plt.get_cmap('gray'))
	

weight_func = np.zeros((n_img,m_img))
# for k in range(3):
for i in range(n_img):
	for j in range(m_img):
		tang_new_x[i][j], tang_new_y[i][j] = T_x(i,j)
		# weight_func[i][j] = T_x(i,j, tang_x, tang_y)

tang_new = tang_new_x**2+tang_new_y**2
tang_new = (tang_new/np.amax(tang_new))
plt.imshow(tang_new)
plt.show()
plt.imsave("abc2.png", tang_new, cmap = plt.get_cmap('gray'))
# plt.imsave("tang_newtest"+str(k)+".png", tang_new, cmap = plt.get_cmap('gray'))

#line drawing
p = 13
q = 5

grad_new_x = tang_new_y					#forming new gradient according to the new tangent
grad_new_y = -tang_new_x

image_pad = np.pad(image, ((p, p), (p,p)), 'constant')
grad_x_p2 = np.pad(grad_new_x, ((p, p), (p,p)), 'constant')                 		#padding the gradient vector with zero
grad_y_p2 = np.pad(grad_new_y, ((p, p), (p,p)), 'constant')

tang_x_p2 = np.pad(tang_new_x, ((p, p), (p,p)), 'constant')
tang_y_p2 = np.pad(tang_new_y, ((p, p), (p,p)), 'constant')

tang_mag2 = np.sqrt(tang_x_p2**2 + tang_y_p2**2)								#taking magnitude of padded tangent vector
tang_x_p2 = np.divide(tang_x_p2, tang_mag2)
tang_y_p2 = np.divide(tang_y_p2, tang_mag2)

tang_x_p2 = np.nan_to_num(tang_x_p2)
tang_y_p2 = np.nan_to_num(tang_y_p2)

gauss_sigma_c = np.zeros(2*q+1)
gauss_sigma_s = np.zeros(2*q+1)
gauss_sigma_m = np.zeros(2*p+1)

sigma_c = 1
sigma_s = 1.6*sigma_c
sigma_m = 3
rho = 0.99

# print(np.sqrt(2*np.pi)*sigma_c)
for i in range(2*q+1):
	gauss_sigma_c[i] = (np.exp(-((-q+i)**2)/(2*sigma_c**2)))/(np.sqrt(2*np.pi)*sigma_c) #gaussian for construction DoG
	# print(gauss_sigma_c[i])
	gauss_sigma_s[i] = (np.exp(-((-q+i)**2)/(2*sigma_s**2)))/(np.sqrt(2*np.pi)*sigma_s)

grad_filter_f = gauss_sigma_c - rho*gauss_sigma_s
	
for i in range(2*p+1):
	gauss_sigma_m[i] = (np.exp(-((-p+i)**2)/(2*sigma_m**2)))/(np.sqrt(2*np.pi)*sigma_m)
	# print(gauss_sigma_m)
F_s = np.zeros((n_img,m_img))
for i in range(n_img):	
	for j in range(m_img):
		# F_s[i][j]
		index_x = np.zeros(2*q+1)						#storing index of point lying on the perpendicular line
		index_y = np.zeros(2*q+1)	
		for t in range(q):
			index_x[t+q+1] = np.int(np.around(i+t + grad_x_p2[i+t][j]))
			index_y[t+q+1] = np.int(np.around(j+t + grad_y_p2[i][j+t]))

			index_x[q-1-t] = np.int(np.around(i-t - grad_x_p2[i-t][j]))
			index_y[q-1-t] = np.int(np.around(j-t - grad_y_p2[i][j-t]))

			index_x[q] = i
			index_y[q] = j

			index_x = index_x.astype(int)
			index_y = index_y.astype(int)

			I_st = image_pad[index_x,index_y]
			I_st = np.array(I_st)

		F_s[i][j] = np.sum(I_st*grad_filter_f)
			# print(index_x, index_y)
F_s_pad = np.pad(F_s, (p,p), 'constant')

# plt.imshow(F_s)
# plt.show()
# plt.imsave("F_s.png", F_s, cmap = plt.get_cmap('gray'))

H_x = np.zeros((n_img,m_img))
for i in range(n_img):	
	for j in range(m_img):
		# F_s[i][j]
		index_x_tang = np.zeros(2*p+1)		#storing index of point lying on the perpendicular line
		index_y_tang = np.zeros(2*p+1)	
		for t in range(p):
			index_x_tang[t+p+1] = np.int(np.around(i+t + tang_x_p2[i+t][j]))
			index_y_tang[t+p+1] = np.int(np.around(j+t + tang_y_p2[i][j+t]))

			index_x_tang[p-1-t] = np.int(np.around(i-t - tang_x_p2[i-t][j]))
			index_y_tang[p-1-t] = np.int(np.around(j-t - tang_y_p2[i][j-t]))

			index_x_tang[p] = i
			index_y_tang[p] = j

			index_x_tang = index_x_tang.astype(int)
			index_y_tang = index_y_tang.astype(int)

			F_s_line = F_s_pad[index_x_tang,index_y_tang]			#storing the value of F(s) from -S to +S
			F_s_line = np.array(F_s_line)

		H_x[i][j] = np.sum(F_s_line*gauss_sigma_m)

# plt.imshow(H_x)
# plt.show()
# plt.imsave("H_x_girl.png", H_x, cmap = plt.get_cmap('gray'))
print(np.max(H_x))
tau = 0.9
H_x[(H_x < 0)] = 0             # & (1+np.tanh(H_x) < tau)] = 0
H_x[ H_x != 0] = 1

print(H_x)
plt.imshow(H_x)
plt.show()
plt.imsave("H_x_girl_q7_3.png", H_x, cmap = plt.get_cmap('gray'))
