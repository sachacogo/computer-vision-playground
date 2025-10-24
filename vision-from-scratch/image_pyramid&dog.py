import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("C:\\Users\\sacha\\OneDrive\\Documents\\Projet\\computer-vision-playground\\vision-from-scratch\\image_eg\\lea.png", cv2.IMREAD_GRAYSCALE)
t = 1
image_b = image.copy()
def impair(size):
        if(size % 2 == 0): 
            return size +1
        else:
            return size 
           
def smooth(image, kernel, h, w): 

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
    s,h,w = D.shape
    D_c = np.zeros_like(D)

    for a in range(1, s-1):
        for b in range(1, h-1):
            for c in range(1, w-1):
                patch = D[a-1:a+2,b-1:b+2,c-1:c+2]
                if np.abs(D[a,b,c]) == np.max(np.abs(patch)):
                    D_c[a,b,c] = D[a,b,c]
                else:
                    D_c[a,b,c] = 0    
                  
    D = D_c                
    return D            

sigma_0 = 0.7
s = 5
lvl = 3

def KernGauss(sigma):

    size = int(sigma*6+1)
    size = impair(size)
    n = size//2

    kernel = np.zeros(size, dtype = np.float32)

    for j in range(size):
        x = j-n
        kernel[j] = np.exp(-(x**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

    kernel /= kernel.sum()     
    return kernel 

def KernGaussGrad(sigma):

    size = int(sigma*3+1)
    size = impair(size)
    n = size//2

    kernelG = np.zeros(size, dtype = np.float32)

    for j in range(size):
        x = j-n
        kernelG[j] = np.exp(-(x**2)/(2*sigma**2))*x/(np.sqrt(2*np.pi)*sigma**3)

    kernelG /= kernelG.sum()     
    return kernelG    

PYRAMID = []
DoG = []

for k in range(lvl):
    DoG_k = []
    DoG_k_n = []
    image = image_b[::t,::t]
    h, w = image.shape
    sigma = sigma_0

    for i in range(s):
        # sigma = 2**(i/s)*sigma_0
        sigma_opti = sigma*np.sqrt(2**(2/s)-1)
        # KerGaus = KernGauss(size, n, sigma)

        if k == i == 0: 
            KerGaus = KernGauss(sigma)
        else:
            KerGaus = KernGauss(sigma_opti)

        sigma = np.sqrt(sigma**2 + sigma_opti**2)

        image = smooth(image, KerGaus, h, w)
        PYRAMID.append(image)

        if(i >= 1):
            DoG_k.append(PYRAMID[-1] - PYRAMID[-2])
            # DoG_k_n = cv2.normalize(DoG_k[i-1], None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imshow("a", DoG_k_n.astype(np.uint8))    
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            continue  

    t = t*2

    DoG_ka = np.array(DoG_k)
    DoG_k_NMS_o = NMS(DoG_k)

    X = np.argwhere(DoG_k_NMS_o != 0)
    DoG_k_NMS = DoG_k_NMS_o.copy()

    # for d in range(1, s-2):
    #     cv2.imshow(f"{d}", DoG_k_NMS_o[d])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    for a,b,c in X:
        
        sigma_s,h_s,w_s = DoG_ka.shape 
        if 1 <= a<sigma_s-1 and 1 <= b < h_s-1 and 1 <= c < w_s-1:

            Dss = DoG_ka[a+1,b,c] - 2*DoG_ka[a,b,c] + DoG_ka[a-1,b,c]
            Dxx = DoG_ka[a,b+1,c] - 2*DoG_ka[a,b,c] + DoG_ka[a,b-1,c]
            Dyy = DoG_ka[a,b,c+1] - 2*DoG_ka[a,b,c] + DoG_ka[a,b,c-1]

            Dsx = (DoG_ka[a+1,b+1,c] - DoG_ka[a+1,b-1,c] - DoG_ka[a-1,b+1,c] + DoG_ka[a-1,b-1,c]) / 4
            Dsy = (DoG_ka[a+1,b,c+1] - DoG_ka[a+1,b,c-1] - DoG_ka[a-1,b,c+1] + DoG_ka[a-1,b,c-1]) / 4
            Dxy = (DoG_ka[a,b+1,c+1] - DoG_ka[a,b+1,c-1] - DoG_ka[a,b-1,c+1] + DoG_ka[a,b-1,c-1]) / 4

            H = np.array([
                [Dss, Dsx, Dsy],
                [Dsx, Dxx, Dxy],
                [Dsy, Dxy, Dyy]
            ], dtype=np.float32)

            grad = np.array([
            (DoG_ka[a+1,b,c] - DoG_ka[a-1,b,c]) / 2,  
            (DoG_ka[a,b+1,c] - DoG_ka[a,b-1,c]) / 2,  
            (DoG_ka[a,b,c+1] - DoG_ka[a,b,c-1]) / 2  
                ]       , dtype=np.float32)
             
            # if np.linalg.det(H) > 1e-10: 
            if np.max(np.abs(H)) > 1e-10:
                try:

                    dX = np.linalg.solve(H, -grad)
    
                except np.linalg.LinAlgError:

                    dX = np.linalg.pinv(H) @ (-grad)
                            
                if np.max(np.abs(dX)) >= 1.0:
                    DoG_k_NMS[a,b,c] = 0
                # print(dX)    

                DoG_k_NMS[a,b,c] = DoG_k_NMS[a,b,c] + 1/2*np.dot(dX, grad)

                if np.abs(DoG_k_NMS[a,b,c]) > 0.003 * np.max(np.abs(DoG_k_NMS)):
                    DoG_k_NMS[a,b,c] = 255
                else:
                    DoG_k_NMS[a,b,c] = 0

                H_2D = np.array([[Dxx, Dxy],
                                 [Dxy, Dyy]])
                
                edgesupress = (np.trace(H_2D))**2/np.linalg.det(H_2D)

                if edgesupress < 12.1:
                    DoG_k_NMS[a,b,c] = 0

                else: continue 
            else:    
                continue
        else:
            DoG_k_NMS[a,b,c] = 0
    


    KP = np.argwhere(DoG_k_NMS != 0)

    for sgma,x,y in KP:
        mirroir = np.pad(DoG_ka, 
                pad_width=((0,0), (8,8), (8,8)), 
                mode='reflect')
        
        patch = mirroir[sgma, x-8:x+8,y-8:y+8]


        p,m=patch.shape
        dxy = KernGaussGrad(1.5*sgma)

        dx = np.zeros_like(patch, dtype=np.float32)
        dy = np.zeros_like(patch, dtype=np.float32)

        for i in range(p):
            dx[i,:] = np.convolve(patch[i,:], dxy, mode="same")
  
        for j in range(m): 
            dy[:,j] = np.convolve(patch[:,j], dxy, mode="same")


        mag = np.sqrt(dx**2+dy**2)
        GRAD_A = np.arctan2(dy, dx)
        angle = np.degrees(GRAD_A)
        angle = (angle + 360) % 360

        nb_bins = 36
        bins = np.linspace(0, 360, nb_bins + 1)

        hist, a = np.histogram(angle, bins=bins, weights=mag)

        # plt.figure(figsize=(6, 4))
        # plt.bar(bins[:-1], hist, width=360/nb_bins, align='edge', edgecolor='black')
        # plt.title("Histogramme d’orientation du gradient")
        # plt.xlabel("Angle (degrés)")
        # plt.ylabel("Amplitude (somme des magnitudes)")
        # plt.show()
    
        angle_max = np.argmax(hist)*10
        new_angle = (angle - angle_max + 360) % 360
        patch4x4 = np.zeros((4,4,4,4), dtype = np.float32)
        dx4 = np.zeros((4,4,4,4), dtype = np.float32)
        dy4 = np.zeros((4,4,4,4), dtype = np.float32)
        ID = np.zeros((4,4,8), dtype = np.float32)


        for i in range(4):
            for j in range(4):

                x = i*4+2
                y = j*4+2

                patch4x4[i, j] = patch[x-2:x+2, y-2:y+2]

                for u in range(1,4-1):
                     for v in range(1,4-1):
                         dx4[i,j,u,:]= (patch4x4[i,j,u+1,:] - patch4x4[i,j,u-1,:] )/2
                         dy4[i,j,:,v]= (patch4x4[i,j,:,v+1] - patch4x4[i,j,:,v-1] )/2

                mag4x4 = np.sqrt(dx4**2+dy4**2)
                angle4x4 = np.arctan2(dy4,dx4)
                angle4 = (360 + np.degrees(angle4x4))%360
                b = np.linspace(0, 360, 9)
                h4x4, a4x4 = np.histogram(angle4,bins=b,weights=mag4x4)

                ID[i,j] = h4x4
        ID/=ID.sum()        


                # plt.figure(figsize=(6, 4))
                # plt.bar(b[:-1], h4x4, width=360/8, align='edge', edgecolor='black')
                # plt.title("Histogramme d’orientation du gradient")
                # plt.xlabel("Angle (degrés)")
                # plt.ylabel("Amplitude (somme des magnitudes)")
                # plt.show()



        # dx4[:,:,1:-1,:]= (patch4x4[:,:,2:,:] - patch4x4[:,:,:-2,:] )/2   
        # dy4[:,:,:,1:-1]= (patch4x4[:,:,:,2:] - patch4x4[:,:,:,:-2] )/2   
        

    DoG.append(DoG_k_NMS)


# for i in range(lvl):
#     for j in range(len(DoG[i])):
#         cv2.imshow(f"{i}{j}", DoG[i][j])    

DoG_f = []

for i in range(lvl):
    tot = DoG[i][0]
    for j in range(1, len(DoG[i])):
        tot += DoG[i][j]
    DoG_f.append(tot)

for i in range(lvl):
    cv2.imshow(f"Octave {i+1}", DoG_f[i].astype(np.uint8))
    
#     nb_keypoints_tot = np.sum(DoG_f[i] > 0)
#     print(f"Octave {i+1} total: {nb_keypoints_tot} keypoints uniques")

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