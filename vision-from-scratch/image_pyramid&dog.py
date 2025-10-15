import cv2
import numpy as np

image = cv2.imread("C:\\Users\\sacha\\OneDrive\\Documents\\Projet\\computer-vision-playground\\vision-from-scratch\\image_eg\\vwv.jpg", cv2.IMREAD_GRAYSCALE)
t = 1
h,w = image.shape

def impair(size):
        if(size % 2 == 0): 
            return size +1
        else:
            return size 
        
       
        
def smooth(image, kernel): 
    Sx = np.zeros_like(image, dtype=np.float32)
    for k in range(h): 
        Sx[k,:] =  np.convolve(image[k,:], kernel, mode ="same")  
    S = np.zeros_like(image, dtype=np.float32)       
    for l in range(w): 
        S[:,l] =  np.convolve(Sx[:,l], kernel, mode ="same")  
    return S;          
    

def NMS(D):
    #[s,x,y]
    D = np.array(D)
    for a in range(1, s-1):
        for b in range(1, h-1):
            for c in range(1, w-1):
                patch = D[a-1:a+2,b-1:b+2,c-1:c+2]
                if D[a,b,c] == np.max(np.abs(patch)):
                    D[a,b,c] = 255
                else:
                    D[a,b,c] = 0    
    return D            


sigma_0 = 1.6
s = 5
lvl = 2

PYRAMID = []
DoG = []

for k in range(lvl):
    DoG_k = []
    image = image[::t,::t]
    h, w = image.shape

    for i in range(s):

        sigma = 2**(i/s)*sigma_0

        size = int(sigma*6+1)
        size = impair(size)
        n = size//2
        kernel = np.zeros(size, dtype = np.float32)
    
        for j in range(size):
            x = j-n
            kernel[j] = np.exp(-(x**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
            
        kernel /= kernel.sum() 

 
        image_smoothed = smooth(image, kernel)


        PYRAMID.append(image_smoothed)

        if(i >= 1):
            DoG_k.append(PYRAMID[-1] - PYRAMID[-2])
            DoG_k[-1] = cv2.normalize(DoG_k[-1], None, 0, 255, cv2.NORM_MINMAX)
        else:
            continue  


  
    t = t*2
    DoG_k = NMS(DoG_k)
    DoG.append(DoG_k)


for i in range(lvl):
    for j in range(s-1):
        cv2.imshow(f"{i},{j}", DoG[i][j].astype(np.uint8))

        


cv2.waitKey(0)
cv2.destroyAllWindows()





  



# pour implémenter en application une bonne image pyramid avec un dog :µ

# Soit S(sigma_i) notre image convoluée à un kernel pour un sigma_i

# on prend notre image I,
# on défini la taille d'un octave : s
# on donne une valeur de sigma 0
# pour chaque sigma tel que
# sigma_i = sigma_0*2^(i/s) <-> sigma_i+1 = sigma_i*2^(1/s) (on pose k = 2^(1/s))
# on convolue notre image I avec notre kernel pour sigma_i
# avec i allant de 0 à i
# (pour diminuer un max la complexité, 2 points : 
# 1/
# pour avoir notre S(sigma_i+1) on convolue S(sigma_i) à G(sigma_i*racine(k^2-1))
# ce qui va nous permettre de ne pas à avoir a stocker tous les sigmas + gagner en complexité car convoluer à un G(sigma_i+1) aurait demandé un kernel plus grand

# 2/
# On peut se dire que si on ne convolue I et G que pour des sigma_i allant de sigma_0 à 2*sigma_0
# on va "louper" plein de blob pour des lissages qui ne seront donc pas pris en compte
# donc on peut se dire qu'on ne s'arrete pas à i = s mais si 
# en revanche au lieu de continuer de augmenter la taille de notre kernel, on va diminuer celle de notre image : 
# exemple
# au lieu de faire I*G(2*sigma_0) on va faire I(x/2, y/2) * G(sigma_0)

# tips : de plus contrairement au LoG, le kernel simple est séparable en 2 vecteurs donc complexité 
# O(K^2WH) -> O(2KWH)
# )

# ensuite, 
# on a vu que le DoG = G(sigma_i+1) - G(sigma_i)
# si on convolue à notre image :
# on va avoir
# S(sigma_i+1) - S(sigma_i)
# conclusion finale : 
# on récupère pour chaque octave nos images : 
# on les soustraits entre elle suivant cette loi : 
# S(sigma_i+1) - S(sigma_i) ≃> Image mettant en valeur les blobs pour sigma_i
# et en sortie on aura moins d'image mais elles représenteront notre approximation en image pyramid de notre scale-space représentation avec les blobs mis en valeur

# Maintenant il faut trouver nos centre de blob
# on applique un NMS sur tout les pixels 
# sur x,y,sigma
# ou on cherche à chaque fois la valeur la plus haute parmi les 26 voisins de chaque pixel 
# (patch de 3x3x3)
# et on récupère donc ces valeurs les plus grandes et ensuite on interpole avec des équations quadratiques qui estimeraient la courbe entre les différents sigma
# et on récupère la pixel value maximum estimée par les courbes à un certain sigma
# et au final on aura une jolie scale-space pyramid sans que ça ne coute trop de mémoire





# donc 
# dX* = -H^(-1)*g
# sert à trouver la position optimal de chaque candidat 
# X+dX
# D(X+dX) sert à calculer le thresholding pour enlever les blobs non pertinant
# et H permet de récupérer les variations à chaque point X+dX et de estimer grâce à ses valeurs propres à quoi correspond quoi