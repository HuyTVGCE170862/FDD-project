#convert raw to grayscale first
from PIL import Image
img = Image.open('F:/Project/FDD/png/new demo.png').convert('LA')
img.save('F:/Project/FDD/grayscale/beta.png')
#grayscale to RGb but t nghĩ m phải crop ra ! vì nó save fig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
img = mpimg.imread('F:/Project/FDD/grayscale/beta.png')
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.savefig('F:/Project/FDD/RGB/beta1.png') 
#well ! sau khi t crop ra thì t sẽ lưu beta1 là beta2 nha !
#but t nhận ra png ko phân tích đc ! nên t đá từ png sang jpg
from PIL import Image
import os, sys
im = Image.open("F:/Project/FDD/RGB/beta2.png")
bg = Image.new("RGB", im.size, (255,255,255))
bg.paste(im,im)
bg.save("F:/Project/FDD/RGB/test.jpg")
#ok ! giờ thì đá từ cái jpg sang diagram
#part1
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('F:/Project/FDD/RGB/test.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
#part2
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('F:/Project/FDD/RGB/test.jpg',0)
plt.hist(img.ravel(),256,[0,256])
plt.show()
#thế là phân tích xong ! :)))