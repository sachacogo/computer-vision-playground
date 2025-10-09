import cv2
import numpy as np

image = cv2.imread("C:\\Users\\sacha\\OneDrive\\Documents\\Projet\\computer-vision-playground\\vision-from-scratch\\image_eg\\camb.jpg", cv2.IMREAD_GRAYSCALE)

h, w = image.shape

s = 7
size = 6*s+1
width = size//2

kernel_2nd = np.zeros((size,size), dtype = np.float32)
kernel_2nd_1D = np.zeros(size, dtype = np.float32)


for i in range(size):
    for j in range(size):
        x = i-width
        y = j-width
        kernel = np.exp(-(x**2+y**2)/(2*s**2))/(2*np.pi*s**2) 
        kernel_2nd[i,j] = ((x**2+y**2) - 2*s**2) / (s**4) * kernel

S_n = cv2.filter2D(image, cv2.CV_32F, kernel_2nd)        

for i in range(size):
    x = i-width
    kernel = np.exp(-x**2/(2*s**2))/(np.sqrt(2*np.pi*s**2)) 
    kernel_2nd_1D[i] = (x**2 - s**2) / (s**4) * kernel
 
S_x = np.zeros_like(image, dtype = np.float32)
S_y = np.zeros_like(image, dtype = np.float32)


for i in range(1, h-1):
    S_x[i,:] = np.convolve(image[i,:], kernel_2nd_1D, mode = "same")

for j in range(1, w-1):
    S_y[:,j] = np.convolve(image[:,j], kernel_2nd_1D, mode = "same")

 #Combinaison
S_n_1D = (S_x + S_y)

# Visualisation : normalisation pour affichage
S_n_1D = (S_n_1D - np.min(S_n_1D)) / (np.max(S_n_1D) - np.min(S_n_1D))*255
S_n = (S_n - np.min(S_n)) / (np.max(S_n) - np.min(S_n))*255
cv2.imshow("LoG1", (S_n).astype(np.uint8))
cv2.imshow("LoG1D", (S_n_1D).astype(np.uint8))


S_n[S_n>200] = 255
S_n[S_n<200] = 0

S_n_1D[S_n_1D>200] = 255
S_n_1D[S_n_1D<200] = 0

cv2.imshow("LoG", (S_n).astype(np.uint8))
cv2.imshow("LoG_1D", (S_n_1D).astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()