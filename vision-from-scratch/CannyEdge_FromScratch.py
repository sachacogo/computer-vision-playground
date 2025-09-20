import cv2
import numpy as np

#thresholds parameters (to adapt depending on the image)
t_max = 90 #treshold max
t_min= 30 #treshold min 

s_x= 1 #sigma for x
s_y= s_x #sigma for y

# Size of the kernel. Without limiting the size, the kernel would consider all pixels in the line, adding a lot of unnecessary computation.
size = 6*s_x +1  #After size = 6*sigma + 1, contributions from further pixels are negligible.


width = size//2 #to center the kernel around 0

image = cv2.imread("----FILE PATH----", cv2.IMREAD_GRAYSCALE) #load image in grayscale
h, w = image.shape #height and width of the image adaptative to any image size

cv2.imshow("i1", image) 

#kernel x (g_s(x)) for the columns
kernelx = np.zeros(size, dtype=np.float32) #kernel's initialization x
# Fill the kernel array with the Gaussian function values
# x = distance from the center
for i in range(size):
    x=i-width #from the left to the right of the kernel (for size = 7, x will take the values -3,-2,-1,0,1,2,3)
    kernelx[i] = np.exp(-x**2/(2*s_x**2))/(np.sqrt(2*np.pi*s_x**2))

# Normalize so that the sum of all elements is 1
kernelx = kernelx/np.sum(kernelx)     
   
#kernel y (g_s(y)) for the rows
kernely = np.zeros(size, dtype=np.float32) #kernel's initialization y
# Fill the kernel array with the Gaussian function values
# y = distance from the center
for j in range(size):
    y=j-width
    kernely[j] = np.exp(-y**2/2*s_y**2)/(np.sqrt(2*np.pi*s_y**2))

# Normalize so that the sum of all elements is 1
kernely = kernely/np.sum(kernely)     
  
#Smooth image = G_s(x) convolved with G_s(y) convolved with I(x,y) (thanks to the separability of the Gaussian kernel)

#convolution of I with g_s(x) 
GxI = np.zeros_like(image, dtype=np.float32) #initialization of the smoothed image on x
for k in range(h):
    GxI[k, :] = np.convolve(image[k,:], kernelx, mode="same") #[k,:] means we take the k-th row of the image

#We now filled GxI with the smoothed image pixel's values on x    

#convolution of GxI with g_s(y)
GxyI = np.zeros_like(image, dtype=np.float32) #initialization of the smoothed image on x and y
for l in range(w):
    GxyI[:,l] = np.convolve(GxI[:,l], kernely, mode="same") #[:,l] means we take the l-th column of GxI

#We now filled GxyI with the smoothed image pixel's values on x and y

dGx = np.zeros_like(image, dtype=np.float32) #initialization of the gradient on x
dGy = np.zeros_like(image, dtype=np.float32) #initialization of the gradient on y
n_Gxy = np.zeros_like(image, dtype=np.float32) #initialization of the norm of the gradient



#computing gradient
for m in range(1, h-1): #avoid the borders
    for n in range(1, w-1):
        dGx[m,n] = (GxyI[m+1,n] - GxyI[m-1,n])/2 #derivative on x (central difference)
        dGy[m,n] = (GxyI[m,n+1] - GxyI[m,n-1])/2 #derivative on y (central difference)
        n_Gxy[m,n] = np.sqrt(dGx[m,n]**2+dGy[m,n]**2) #norm of the gradient

n_Gxy_display = cv2.normalize(n_Gxy, None, 0, 255, cv2.NORM_MINMAX) #normalization for display purpose (8-bit image)

#non-maximum suppression
for u in range(1, h-1): #avoid the borders 
    for v in range(1, w-1):
        gradA = np.arctan2(dGy[u,v], dGx[u,v]) #angle of the gradient
        angle_deg = np.degrees(gradA) #conversion in degrees
        if (abs(angle_deg) < 22.5) or (abs(angle_deg-180)<22.5): #horizontal edge
            if n_Gxy_display[u,v] < n_Gxy_display[u-1,v] or n_Gxy_display[u,v] < n_Gxy_display[u+1,v]: #compare to the pixel above and below
                n_Gxy_display[u,v] = 0
        elif (abs(angle_deg-45) < 22.5) or (abs(angle_deg-225)<22.5): #diagonal edge (45 degrees)
            if n_Gxy_display[u,v] < n_Gxy_display[u+1,v-1] or n_Gxy_display[u,v] < n_Gxy_display[u-1,v+1]:
                n_Gxy_display[u,v] = 0      
        elif (abs(angle_deg-90) < 22.5) or (abs(angle_deg-270)<22.5): #vertical edge
            if n_Gxy_display[u,v] < n_Gxy_display[u,v-1] or n_Gxy_display[u,v] < n_Gxy_display[u,v+1]:
                n_Gxy_display[u,v] = 0            
        elif (abs(angle_deg-135) < 22.5) or (abs(angle_deg-315)<22.5): #diagonal edge (135 degrees)
            if n_Gxy_display[u,v] < n_Gxy_display[u-1,v-1] or n_Gxy_display[u,v] < n_Gxy_display[u+1,v+1]:
                n_Gxy_display[u,v] = 0  


#thresholding 
for a in range(h): 
    for b in range(w):
        if (n_Gxy_display[a,b] > t_max): #strong edge
            n_Gxy_display[a,b]=255
        elif (n_Gxy_display[a,b] < t_max) and (n_Gxy_display[a,b] > t_min): #weak edge
            n_Gxy_display[a,b]=128
        else: #not an edge
            n_Gxy_display[a,b]=0


Gxy = n_Gxy_display.copy() #creating a copy to avoid updating a weak edge to a strong edge and then using this new value to update another weak edge next to it
for g in range(h):
    for t in range(w):
        if (n_Gxy_display[g,t] == 128) and (n_Gxy_display[g+1,t+1] == 255 or n_Gxy_display[g,t+1] == 255 or n_Gxy_display[g-1,t+1] == 255 or n_Gxy_display[g-1,t] == 255 or n_Gxy_display[g-1,t-1] == 255 or n_Gxy_display[g,t-1] == 255 or n_Gxy_display[g+1,t-1] == 255 or n_Gxy_display[g+1,t] == 255): #if a weak edge has at least one strong edge in its 8-neighborhood
            Gxy[g,t] = 255 #update the weak edge to a strong edge
        else: #if not, suppress the weak edge
            Gxy[g,t] = 0
n_Gxy_display = Gxy #update n_Gxy_display with the final result after hysteresis

cv2.imshow("image", n_Gxy_display.astype(np.uint8)) #display the final result
cv2.waitKey(0)
cv2.destroyAllWindows()