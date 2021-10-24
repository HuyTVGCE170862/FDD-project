------------------------20-8-2020-1:10A.m---------
import numpy as np
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
image_gs = imread('/home/jun/Downloads/dog.png', as_gray=True)
fig, ax = plt.subplots(figsize=(9, 16))
imshow(image_gs, ax=ax)
ax.set_title('Grayscale image')
ax.axis('off');
image_rgb = imread('/home/jun/Downloads/dog.png')
fig, ax = plt.subplots(1, 4, figsize = (18, 30))
ax[0].imshow(image_rgb/255.0) 
ax[0].axis('off')
ax[0].set_title('original RGB')
for i, lab in enumerate(['R','G','B'], 1):
    temp = np.zeros(image_rgb.shape)
    temp[:,:,i - 1] = image_rgb[:,:,i - 1]
    ax[i].imshow(temp/255.0) 
    ax[i].axis("off")
    ax[i].set_title(lab)


plt.show() #oke ! that can fucking compile out RGB result (2 enters)
------------------------------------------
#Alternatively, we can plot the separate color channels as follows:
fig, ax = plt.subplots(1, 4, figsize = (18, 30))
ax[0].imshow(image_rgb) 
ax[0].axis('off')
ax[0].set_title('original RGB')
for i, cmap in enumerate(['Reds','Greens','Blues']):
    ax[i+1].imshow(image_rgb[:,:,i], cmap=cmap) 
    ax[i+1].axis('off')
    ax[i+1].set_title(cmap[0])


plt.show()#Run oke !
------------------------------------------
#then i'll test with LAB color
#Value
#L: the lightness on a scale from 0 (black) to 100 (white), which in fact is a grayscale image
#a: green-red color spectrum, with values ranging from -128 (green) to 127 (red)
#b: blue-yellow color spectrum, with values ranging from -128 (blue) to 127 (yellow)
#maybe.... ????
image_lab = rgb2lab(image_rgb / 255)
fig, ax = plt.subplots(1, 4, figsize = (18, 30))
ax[0].imshow(image_lab) 
ax[0].axis('off')
ax[0].set_title('Lab')
for i, col in enumerate(['L', 'a', 'b'], 1):
    imshow(image_lab[:, :, i-1], ax=ax[i])
    ax[i].axis('off')
    ax[i].set_title(col)


fig.show()
#run oke ! but "UserWarning: Float image out of standard range; displaying image with stretched contrast." maybe i'll use smaller pic !
#i'll try with second fig !
image_lab_scaled = (image_lab + [0, 128, 128]) / [100, 255, 255]
fig, ax = plt.subplots(1, 4, figsize = (18, 30))
ax[0].imshow(image_lab_scaled) 
ax[0].axis('off')
ax[0].set_title('Lab scaled')
for i, col in enumerate(['L', 'a', 'b'], 1):
    imshow(image_lab_scaled[:, :, i-1], ax=ax[i])
    ax[i].axis('off')
    ax[i].set_title(col)
    

fig.show() #it's absolutely look fucking good ! it's just a lil prob with the scale of my sample ! not a holycrap ! =)))
#oke ! maybe i'll try with fucking last attemp before i go to bed ! so fucking tired !
fig, ax = plt.subplots(1, 4, figsize = (18, 30))
ax[0].imshow(image_lab_scaled) 
ax[0].axis('off')
ax[0].set_title('Lab scaled')
imshow(image_lab_scaled[:,:,0], ax=ax[1]) 
ax[1].axis('off')
ax[1].set_title('L')
ax[2].imshow(image_lab_scaled[:,:,1], cmap='RdYlGn_r') 
ax[2].axis('off')
ax[2].set_title('a')
ax[3].imshow(image_lab_scaled[:,:,2], cmap='YlGnBu_r') 
ax[3].axis('off')
ax[3].set_title('b')
    
plt.show()
#oke !This time the results are satisfactory. We can clearly distinguish different colors in the a and b layers ! 