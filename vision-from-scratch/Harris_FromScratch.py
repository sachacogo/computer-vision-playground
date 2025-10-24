import cv2
import numpy as np

k=3 #size of our patches
alpha = 0.04 #R coefficient (usually 0.04 to 0.06)

s = 1 #our second smoothing (not the sobel's one) sigma (1 because our patches is also small)
size = 6*s+1 #function I found on internet correling the sigma's value to his size (in a discrete signal it's useless to let the kernel have infinite size because after a certain value he's very close to 0 so it's useless calculations)
width = size//2 #defining the center of the kernel

image = cv2.imread("C:\\Users\\sacha\\OneDrive\\Documents\\Projet\\computer-vision-playground\\vision-from-scratch\\image_eg\\lea.png", cv2.IMREAD_GRAYSCALE) #get image with cv2 functions 
h, w = image.shape #get shape image

dxy = np.array([1,0,-1], np.float32) /2 #sobel matrix but separated in 2 vectors to theorically reduce the complexity from 2K^2WH
gxys = np.array([1,2,1], np.float32) /4 #to 4KWH (maybe not useful here because K is small but I like it this way still)
#d for derivative and g for smoothing *

#sobel implementation 

Ig_x = np.zeros_like(image, dtype = np.float32) #creating matrix full of 0 ready to be filled with interesting values
Ig_y = np.zeros_like(image, dtype = np.float32) #same size as the image

kernel = np.zeros(size, dtype = np.float32) #kernel's array

for i in range(size):
    x = i - width #so that "0" is the center of the kernel's array
    kernel[i] = np.exp(-x**2/(2*s**2))/(np.sqrt(2*np.pi*s**2)) #from  scratch kernel's implementation with the real kernel's formula

kernel /= np.sum(kernel)

#now we're gonna smooth are image and compute gradient to have a smoothed image (the order is not important since the convolution is commutative):
# e.g : dS/dx =  dx * gy * I(x,y) 
#the sobel implementation force the derivative and the smoothing to be orthogonal so the smoothing does not affect the real pixel value variation and still reduce the noise

for i in range(1, h-1):
    Ig_x[i,:] = np.convolve(image[i,:], gxys, mode = "same") #gx * I(x,y)

for j in range(1, w-1):
    Ig_y[:,j] = np.convolve(image[:,j], gxys, mode = "same")  #gy * I(x,y)

Sx = np.zeros_like(image, np.float32)
Sy = np.zeros_like(image, np.float32)


for i in range(1, h-1):
    Sx[i,:] = np.convolve(Ig_y[i,:], dxy,  mode = "same") #dx * (y * I(x,y))

for j in range(1, w-1):
    Sy[:,j] = np.convolve(Ig_x[:,j], dxy , mode = "same") #dy * (x * I(x,y)  

#Sn = np.sqrt(Sx**2 + Sy**2) 
#cv2.imshow("Sn", Sn.astype(np.uint8))

Ix = np.zeros_like(image, dtype = np.float32)
R = np.zeros_like(image, dtype = np.float32)

#now we're starting the real stuff
#we're going to create the matrix M that allows us to determine if the Error Value is big enough to be a edge or not
for x in range(1, h-1):
    for y in range(1, w-1):
        #create a patch (3x3) with the smoothed + gradient computed image on x & y 
        Ix = Sx[x-1:x+2, y-1:y+2]
        Iy = Sy[x-1:x+2, y-1:y+2]

        #for x,y in the patch do the sum :
        Ix2 = np.sum(Ix**2)
        Iy2 = np.sum(Iy**2)
        IyIx = np.sum(Ix*Iy)

        #finally the M matrix
        M = np.array([[Ix2, IyIx],
                      [IyIx, Iy2]], dtype = np.float32)  
                
        #thanks to M we're able to use this formula to analyze if a patch is an edge or not
        R[x,y] = np.linalg.det(M) - alpha * (np.trace(M)**2)
        #R[x,y] = Ix2*Iy2-IyIx**2 - alpha * (Ix2+Iy2)

     
#smoothing the R values one last time
R_s = np.zeros_like(R)
for i in range(h):
    R_s[i,:] = np.convolve(R[i,:], kernel, mode="same")
for j in range(w):
    R_s[:,j] = np.convolve(R_s[:,j], kernel, mode="same")
R = R_s

#thresholding delete every pixel not big enough (we only want the main edges)
R[R < 0.05 * np.max(R)] = 0

R_p = np.zeros_like(R, dtype = np.float32)

for x in range(1, h-1):
    for y in range(1, w-1):
        patch = R[x-1:x+2, y-1:y+2] #for every patch in the matrix we're looking for the biggest pixel and delete all the other one (NON MAXIMAL SUPPRESSOR)
        if R[x,y] == np.max(patch):
            R_p[x,y] = R[x,y]
        else:
            R_p[x,y] = 0

R = R_p

R = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
   
cv2.imshow("Corner1", R.astype(np.uint8))
cv2.imshow("image principale", image)

cv2.waitKey(0)
cv2.destroyAllWindows()