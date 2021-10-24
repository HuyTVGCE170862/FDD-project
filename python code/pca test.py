from PIL import Image
import os, numpy, PIL 
from numpy import *
from pylab import *
import pca 
import pickle
allfiles=os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if filenam[-4:] in [".png"]]
im = array(Image.open(imlist[0]))
m,n = im.shape[0,2]
imnbr = list(imlist)
immatrix = asarray([array(Image.open(im)).flatten() for im in imlist], 'f')
V,S,immean = pca.pca(immatrix)
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n)
for i in range (7):
    subplot(2,4,i+2)
    imshow(V[i].reshape(m,n))

show()

f = open('F:/coding/Project/FDD/pca.pkl')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close