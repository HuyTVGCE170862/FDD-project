import matplotlib.pyplot as plt
import cv2
import numpy as np

image = plt.imread('../heatmapwms.png')

height, width = image.shape[0:2]
for i in range(0, height):  
    for j in range(0, width):
        if (image[i][j][3] <= .34 or (
                (image[i][j][2] * 255 > 170) and (image[i][j][1] * 255 > 150) and (image[i][j][0] * 255 > 150))):
            image[i][j] = 0

kernel = np.ones((3, 3), np.float32) / 9
image = cv2.filter2D(image, -1, kernel)

for i in range(0, height):  
    for j in range(0, width):
        if (image[i][j][3] <= .30 or (
                (image[i][j][2] * 255 > 170) and (image[i][j][1] * 255 > 150) and (image[i][j][0] * 255 > 150))):
            image[i][j] = 0

kernel = np.ones((3, 3), np.float32) / 9
image = cv2.filter2D(image, -1, kernel)

plt.imshow(image)
plt.savefig("filename.png")
plt.show()
#extract value of pixel in image !
from PIL import Image
import numpy as np
import cv2
def initialization_rotate(path):
    global h,w,image
    img4 = np.array(Image.open(path).convert('L'))
    img3 = img4.transpose(1,0)
    img2 = img3[::-1,::1]
    img = img2[400:1000,1:248]
    h, w = img.shape

path = 'F:/coding/Project/FDD/neo1.png'

def opening(binary):
    opened = np.zeros_like(binary)              
    for j in range(1,w-1):
        for i in range(1,h-1):
            if binary[i][j]> 100:
                n1 = binary[i-1][j-1]
                n2 = binary[i-1][j]
                n3 = binary[i-1][j+1]
                n4 = binary[i][j-1]
                n5 = binary[i][j+1]
                n6 = binary[i+1][j-1]
                n7 = binary[i+1][j]
                n8 = binary[i+1][j+1]
                sum8 = int(n1) + int(n2) + int(n3) + int(n4) + int(n5) + int(n6) + int(n7) + int(n8)
                if sum8 < 1000:
                    opened[i][j] = 0
                else:
                    opened[i][j] = 255
            else:
                pass
    return opened    


edge = np.zeros_like(img)


# Find the max pixel value and extract the postion
for j in range(w-1):
    ys = [0]
    ymax = []
    for i in range(h-1):
         if img[i][j] > 100:
            ys.append(i)
        else:
            pass
    ymax = np.amax(ys)
    edge[ymax][j] = 255


cv2.namedWindow('edge')

while(True):
    cv2.imshow('edge',edge)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()