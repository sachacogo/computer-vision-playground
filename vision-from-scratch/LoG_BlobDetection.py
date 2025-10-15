import cv2
import numpy as np

image = cv2.imread("C:\\Users\\sacha\\OneDrive\\Documents\\Projet\\computer-vision-playground\\vision-from-scratch\\image_eg\\camb.jpg", cv2.IMREAD_GRAYSCALE)

h, w = image.shape

t = 1

def impair(size):
        if(size % 2 == 0): 
            return size +1
        else:
            return size 

# h = h//t
# w = w//t

# image = cv2.resize(image, (w, h))
# my intuitive way of subsampling theoricaly work but add more complexity (calculate the average pixel value with his surroundings for each pixel) but has less risk of losing patern (aliasing)

s = 1.5
size = int(6*s+1)
size = impair(size)
width = size//2

kernel_2nd = np.zeros((size,size), dtype = np.float32)
kernel_2nd_1D = np.zeros(size, dtype = np.float32)


for i in range(size):
    for j in range(size):
        x = i-width
        y = j-width
        kernel = np.exp(-(x**2+y**2)/(2*s**2))/(2*np.pi*s**2) 
        kernel_2nd[i,j] = ((x**2+y**2) - 2*s**2) / (s**4) * kernel

image = image[::t, ::t]     #best way of implementing the subsampling with the most efficient complexity
S_n = cv2.filter2D(image, cv2.CV_32F, kernel_2nd)   
 


# for i in range(size):
#     x = i-width
#     kernel = np.exp(-x**2/(2*s**2))/(np.sqrt(2*np.pi*s**2)) 
#     kernel_2nd_1D[i] = (x**2 - s**2) / (s**4) * kernel
 
# S_x = np.zeros_like(image, dtype = np.float32)
# S_y = np.zeros_like(image, dtype = np.float32)


# for i in range(1, h-1):
#     S_x[i,:] = np.convolve(image[i,:], kernel_2nd_1D, mode = "same")

# for j in range(1, w-1):
#     S_y[:,j] = np.convolve(image[:,j], kernel_2nd_1D, mode = "same")

#  #Combinaison
# S_n_1D = (S_x + S_y)

# # Visualisation : normalisation pour affichage
# S_n_1D = (S_n_1D - np.min(S_n_1D)) / (np.max(S_n_1D) - np.min(S_n_1D))*255
S_n = (S_n - np.min(S_n)) / (np.max(S_n) - np.min(S_n))*255
cv2.imshow("LoG1", (S_n).astype(np.uint8))
# cv2.imshow("LoG1D", (S_n_1D).astype(np.uint8))

S_n[S_n>200] = 255
S_n[S_n<55] = 255 
S_n[(S_n >= 55) & (S_n <= 200)] = 0




# S_n_1D[S_n_1D>200] = 255
# S_n_1D[S_n_1D<55] = 255 
# S_n_1D[(S_n_1D >= 55) & (S_n_1D <= 200)] = 0


# for i in range(len(S_n_1D)):
#     S_n_1D[i] = [255 if(x>200 and x<10)  else 0 if(x<200 and x>10) else x for x in S_n_1D[i]]

cv2.imshow("LoG", (S_n).astype(np.uint8))
# cv2.imshow("LoG_1D", (S_n_1D).astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()


