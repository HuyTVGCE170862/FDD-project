from PIL import Image
im = Image.open("F:/Project/FDD/city.jpg")
im_L = im.convert("L")
im_L.save("F:/Project/FDD/city2.jpg")
from PIL import Image
from collections import defaultdict
import pprint

img = Image.open("F;/Project/FDD/city2.jpg")
rgbimg = Image.new("RGBA", img.size)
rgbimg.paste(img)

found_colors = defaultdict(int)
for x in range(0, rgbimg.size[0]):
    for y in range(0, rgbimg.size[1]):
        pix_val = rgbimg.getpixel((x, y))
        found_colors[pix_val] += 1 

rgbimg.save('F:/Project/FDD/city_rgb.jpg')
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('F:/Project/FDD/city_rgb.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()