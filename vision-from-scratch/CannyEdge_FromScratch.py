import cv2
import numpy as np

#thresholds parameters (to adapt depending on the image)
t_max = 50 #treshold max
t_min= 10 #treshold min 

s_x= 1 #sigma for x
s_y= s_x #sigma for y

# Size of the kernel. Without limiting the size, the kernel would consider all pixels in the line, adding a lot of unnecessary computation.
size = int(6*s_x+1)  #After size = 6*sigma + 1, contributions from further pixels are negligible.

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
width = size//2 #to center the kernel around 0

#image = cv2.imread("---PICTURE'S PATH", cv2.IMREAD_GRAYSCALE) #load image in grayscale

h, w = image.shape #height and width of the image adaptative to any image size

cv2.imshow("i1", image) 

#sobel approxmation implementation 
d = np.array([-1,0,1], np.float32) / 2 #approximation of the derivative
g = np.array([1,2,1], np.float32) / 4 #approximation of the kernel

pre_sobx = np.zeros_like(image, dtype=np.float32)
pre_soby = np.zeros_like(image, dtype=np.float32)

sobelx = np.zeros_like(image, dtype=np.float32)
sobely = np.zeros_like(image, dtype=np.float32)

for j in range(w):
    pre_sobx[:,j] = np.convolve(image[:,j], g, mode="same")

for i in range(h):
    sobelx[i,:] = np.convolve(pre_sobx[i,:], d, mode="same")

for i in range(h):
    pre_soby[i,:] = np.convolve(image[i,:], g, mode="same")

for j in range(w):
    sobely[:,j] = np.convolve(pre_soby[:,j], d, mode="same")  

sob = np.sqrt(sobelx**2 + sobely**2)  

for u in range(1, h-1): #avoid the borders 
    for v in range(1, w-1):
        gradA = np.arctan2(sobely[u,v], sobelx[u,v]) #angle of the gradient
        angle_deg = np.degrees(gradA) #conversion in degrees
        if (abs(angle_deg) < 22.5) or (abs(angle_deg-180)<22.5): #horizontal edge
            if sob[u,v] < sob[u-1,v] or sob[u,v] < sob[u+1,v]: #compare to the pixel above and below
                sob[u,v] = 0
        elif (abs(angle_deg-45) < 22.5) or (abs(angle_deg-225)<22.5): #diagonal edge (45 degrees)
            if sob[u,v] < sob[u+1,v-1] or sob[u,v] < sob[u-1,v+1]:
                sob[u,v] = 0      
        elif (abs(angle_deg-90) < 22.5) or (abs(angle_deg-270)<22.5): #vertical edge
            if sob[u,v] < sob[u,v-1] or sob[u,v] < sob[u,v+1]:
                sob[u,v] = 0            
        elif (abs(angle_deg-135) < 22.5) or (abs(angle_deg-315)<22.5): #diagonal edge (135 degrees)
            if sob[u,v] < sob[u-1,v-1] or sob[u,v] < sob[u+1,v+1]:
                sob[u,v] = 0

# Thresholding avec hysteresis (comme Canny)
for a in range(h): 
    for b in range(w):
        if (sob[a,b] > t_max): #strong edge
            sob[a,b] = 255
        elif (sob[a,b] < t_max) and (sob[a,b] > t_min): #weak edge
            sob[a,b] = 128
        else: #not an edge
            sob[a,b] = 0

# Hysteresis - connexion des contours
Gxy = sob.copy() #creating a copy to avoid updating a weak edge to a strong edge and then using this new value to update another weak edge next to it

for g in range(1, h-1): # Éviter les bords pour les vérifications de voisinage
    for t in range(1, w-1):
        if (sob[g,t] == 128): # weak edge
            # Vérifier les 8 voisins
            if (sob[g+1,t+1] == 255 or sob[g,t+1] == 255 or sob[g-1,t+1] == 255 or 
                sob[g-1,t] == 255 or sob[g-1,t-1] == 255 or sob[g,t-1] == 255 or 
                sob[g+1,t-1] == 255 or sob[g+1,t] == 255): 
                Gxy[g,t] = 255 #update the weak edge to a strong edge
            else: #if not, suppress the weak edge
                Gxy[g,t] = 0

sob = Gxy #update sob with the final result after hysteresis

# Normalisation pour l'affichage
sob_display = cv2.normalize(sob, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("i6", sob_display.astype(np.uint8)) #display the final result

#kernel x (g_s(x)) for the columns
kernel = np.zeros(size, dtype=np.float32) #kernel's initialization x
# Fill the kernel array with the Gaussian function values
# x = distance from the center
for i in range(size):
    x=i-width #from the left to the right of the kernel (for size = 7, x will take the values -3,-2,-1,0,1,2,3)
    kernel[i] = np.exp(-x**2/(2*s_x**2))/(np.sqrt(2*np.pi*s_x**2))

# Normalize so that the sum of all elements is 1
kernel = kernel/np.sum(kernel)     
   
#Smooth image = G_s(x) convolved with G_s(y) convolved with I(x,y) (thanks to the separability of the Gaussian kernel)

#convolution of I with g_s(x) 
GxI = np.zeros_like(image, dtype=np.float32) #initialization of the smoothed image on x
for k in range(h):
    GxI[k, :] = np.convolve(image[k,:], kernel, mode="same") #[k,:] means we take the k-th row of the image

#We now filled GxI with the smoothed image pixel's values on x    

#convolution of GxI with g_s(y)
GxyI = np.zeros_like(image, dtype=np.float32) #initialization of the smoothed image on x and y
for l in range(w):
    GxyI[:,l] = np.convolve(GxI[:,l], kernel, mode="same") #[:,l] means we take the l-th column of GxI

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
cv2.imshow("i8", n_Gxy_display.astype(np.uint8))
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
#cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()