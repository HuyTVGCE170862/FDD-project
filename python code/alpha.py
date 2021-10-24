#convert jpg to png
from PIL import Image
im = Image.open('F:/Project/FDD/test.jpg')
im.save('F:/Project/FDD/test2.png') #nhớ cái chuyển
#convert png into fucking grayscale
from PIL import Image
im = Image.open("F:/Project/FDD/test2.png")
im_L = im.convert("L")
im_L.save("F:/Project/FDD/test2_grayscale.png")
#convert greyscale into RGB 
from PIL import Image
from collections import defaultdict
import pprint

img = Image.open("F:/Project/FDD/test2_grayscale.png")
rgbimg = Image.new("RGBA", img.size)
rgbimg.paste(img)

found_colors = defaultdict(int)
for x in range(0, rgbimg.size[0]):
    for y in range(0, rgbimg.size[1]):
        pix_val = rgbimg.getpixel((x, y))
        found_colors[pix_val] += 1 

rgbimg.save('F:/Project/FDD/test2_rgb.png')
#DO THAT SHIT
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import timeit
from PIL import Image
def pil_test():
    cm_hot = mpl.cm.get_cmap('CMRmap')
    img_src = Image.open('F:/Project/FDD/test2_rgb.png').convert('L')
    img_src.thumbnail((512,512))
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    im.save('F:/Project/FDD/test2_2.png')

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

def plt_test():
    img_src = mpimg.imread('F:/Project/FDD/test2_rgb.png')
    im = rgb2gray(img_src)
    f = plt.figure(figsize=(4, 4), dpi=128)
    plt.axis('off')
    plt.imshow(im, cmap='CMRmap')
    plt.savefig('F:/Project/FDD/test2_3.png', dpi=f.dpi)
    plt.close()

t = timeit.timeit(pil_test, number=30)
print('PIL: %s' % t)
t = timeit.timeit(plt_test, number=30)
print('PLT: %s' % t)
#then make it into grayscale then to RGB again 
from PIL import Image
im = Image.open("F:/Project/FDD/test2_3.png")
im_L = im.convert("L")
im_L.save("F:/Project/FDD/test3_grayscale.png")
from PIL import Image
from collections import defaultdict
import pprint

img = Image.open("F:/Project/FDD/test3_grayscale.png")
rgbimg = Image.new("RGBA", img.size)
rgbimg.paste(img)

found_colors = defaultdict(int)
for x in range(0, rgbimg.size[0]):
    for y in range(0, rgbimg.size[1]):
        pix_val = rgbimg.getpixel((x, y))
        found_colors[pix_val] += 1 

rgbimg.save('F:/Project/FDD/test3_rgb.png')
#RGB to LAB
import numpy as np
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
image_gs = imread('F:/Project/FDD/test3_rgb.png', as_gray=True)
fig, ax = plt.subplots(figsize=(9, 16))
imshow(image_gs, ax=ax)
ax.set_title('Grayscale image')
ax.axis('off');
image_lab = rgb2lab(image_rgb / 255) #print data
fig, ax = plt.subplots(1, 4, figsize = (18, 30))
ax[0].imshow(image_lab) 
ax[0].axis('off')
ax[0].set_title('Lab')
for i, col in enumerate(['L', 'a', 'b'], 1):
    imshow(image_lab[:, :, i-1], ax=ax[i])
    ax[i].axis('off')
    ax[i].set_title(col)
fig.show()
#crop it then save with name "lab"
#F:/Project/FDD/lab.png
#Noise Removal
import matplotlib.pyplot as plt
import cv2
import numpy as np
image = plt.imread('F:/Project/FDD/lab.png')
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
plt.savefig("F:/Project/FDD/lab2.png")
plt.show()
#maybe we can test some dimension of color map ? (untested)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

node_coordinate =   {1: [0.0, 1.0], 2: [0.0, 0.0], 3: [4.018905, 0.87781],
                        4: [3.978008, -0.1229], 5: [1.983549, -0.038322],
                        6: [2.013683, 0.958586], 7: [3.018193, 0.922264],
                        8: [2.979695, -0.079299], 9: [1.0070439, 0.989987],
                        10: [0.9909098, -0.014787999999999999]}
element_stress =        {1: 0.2572e+01, 2: 0.8214e+00, 3: 0.5689e+01,
                        4: -0.8214e+00, 5: -0.2572e+01, 6: -0.4292e+01,
                        7: 0.4292e+01, 8: -0.5689e+01}

n = len(element_stress.keys())
x = np.empty(n)
y = np.empty(n)
d = np.empty(n)
>>>
for i in element_stress.keys():
        x[i-1] = node_coordinate[i][0]
        y[i-1] = node_coordinate[i][1]
        d[i-1] = element_stress[i]

mask = np.logical_or(x < 1.e20, y < 1.e20)
x = np.compress(mask, x)
y = np.compress(mask, y)
triang = tri.Triangulation(x, y)
cmap = mpl.cm.jet
fig = plt.figure(figsize=(80, 40))
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
cax = ax1.tricontourf(triang, d, cmap=cmap)
fig.colorbar(cax)
<matplotlib.colorbar.Colorbar object at 0x000001C79194E460>
plt.show()
#oke ! i think i'll create more sample to train and get PCA
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
def data_aug(img = img):
	mu = 0
	sigma = 0.1
	feature_vec=np.matrix(evecs_mat)

	# 3 x 1 scaled eigenvalue matrix
	se = np.zeros((3,1))
	se[0][0] = np.random.normal(mu, sigma)*evals[0]
	se[1][0] = np.random.normal(mu, sigma)*evals[1]
	se[2][0] = np.random.normal(mu, sigma)*evals[2]
	se = np.matrix(se)
	val = feature_vec*se

	# Parse through every pixel value.
	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			# Parse through every dimension.
			for k in xrange(img.shape[2]):
				img[i,j,k] = float(img[i,j,k]) + float(val[k])

imnames = ['n00.jpg','n01.jpg','n02.jpg','n03.jpg','n04.jpg','n05.jpg']
#load list of images
imlist = (io.imread_collection(imnames))
res = np.zeros(shape=(1,3))
for i in range(len(imlist)):
	# re-size all images to 256 x 256 x 3
	m=transform.resize(imlist[i],(256,256,3))
	# re-shape to make list of RGB vectors.
	arr=m.reshape((256*256),3)
	# consolidate RGB vectors of all images
	res = np.concatenate((res,arr),axis=0)
res = np.delete(res, (0), axis=0)
# subtracting the mean from each dimension
m = res.mean(axis = 0)
res = res - m
R = np.cov(res, rowvar=False)
print R
from numpy import linalg as LA
evals, evecs = LA.eigh(R)
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx]
evecs = evecs[:, :3]
m = np.dot(evecs.T, res.T).T
img = imlist[0]/255.0
data_aug(img)
plt.imshow(img)
#boundary extraction after pca
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread as imread
plt.figure(1)
img_DR = cv2.imread('F:/Project/FDD/n01.png',0)
edges_DR = cv2.Canny(img_DR,20,40)
plt.subplot(121),plt.imshow(img_DR)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges_DR,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

#tach pixel
import plotly.graph_objects as go 
import numpy as np
  
feature_x = np.arange(0, 50, 2) 
feature_y = np.arange(0, 50, 3) 
  
# Creating 2-D grid of features 
[X, Y] = np.meshgrid(feature_x, feature_y) 
  
Z = np.cos(X / 2) + np.sin(Y / 4) 
  
fig = go.Figure(data=go.Heatmap( 
  x=feature_x, y=feature_y, z=Z,)) 
  
fig.update_layout( 
    margin=dict(t=200, r=200, b=200, l=200), 
    showlegend=False, 
    width=700, height=700, 
    autosize=False) 
  
  
fig.show()

#test demo 12/26/2020/5:45P.M
#convert jpg to png
from PIL import Image
im = Image.open('F:/coding/Project/FDD/neo.jpg')
im.save('F:/coding/Project/FDD/neo1.png') #nhớ cái chuyển
#convert png into fucking grayscale
from PIL import Image
im = Image.open("F:/coding/Project/FDD/neo1.png")
im_L = im.convert("L")
im_L.save("F:/coding/Project/FDD/neo2.png")
#convert greyscale into RGB 
from PIL import Image
from collections import defaultdict
import pprint

img = Image.open("F:/coding/Project/FDD/neo2.png")
rgbimg = Image.new("RGBA", img.size)
rgbimg.paste(img)

found_colors = defaultdict(int)
for x in range(0, rgbimg.size[0]):
    for y in range(0, rgbimg.size[1]):
        pix_val = rgbimg.getpixel((x, y))
        found_colors[pix_val] += 1 

rgbimg.save('F:/coding/Project/FDD/neo3.png')
#DO THAT SHIT
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import timeit
from PIL import Image
def pil_test():
    cm_hot = mpl.cm.get_cmap('CMRmap')
    img_src = Image.open('F:/coding/Project/FDD/neo3.png').convert('L')
    img_src.thumbnail((512,512))
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    im.save('F:/coding/Project/FDD/neo4.png')

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

def plt_test():
    img_src = mpimg.imread('F:/coding/Project/FDD/neo4.png')
    im = rgb2gray(img_src)
    f = plt.figure(figsize=(4, 4), dpi=128)
    plt.axis('off')
    plt.imshow(im, cmap='CMRmap')
    plt.savefig('F:/coding/Project/FDD/neo5.png', dpi=f.dpi)
    plt.close()

t = timeit.timeit(pil_test, number=30)
print('PIL: %s' % t)
t = timeit.timeit(plt_test, number=30)
print('PLT: %s' % t)
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
from past.builtins import xrange
img_file = Image.open("F:/coding/Project/FDD/neo5.png")
img = img_file.load()
[xs, ys] = img_file.size
max_intensity = 100
hues = {}
for x in xrange(0, xs):
  for y in xrange(0, ys):
    [r, g, b] = img[x, y]
    r /= 255.0
    g /= 255.0
    b /= 255.0
    [h, s, v] = colorsys.rgb_to_hsv(r, g, b)
    if h not in hues:
      hues[h] = {}
    if v not in hues[h]:
      hues[h][v] = 1
    else:
      if hues[h][v] < max_intensity:
        hues[h][v] += 1

h_ = []
v_ = []
i = []
colours = []
for h in hues:
  for v in hues[h]:
    h_.append(h)
    v_.append(v)
    i.append(hues[h][v])
    [r, g, b] = colorsys.hsv_to_rgb(h, 1, v)
    colours.append([r, g, b])

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(h_, v_, i, s=5, c=colours, lw=0)
ax.set_xlabel('Hue')
ax.set_ylabel('Value')
ax.set_zlabel('Intensity')
fig.add_axes(ax)
plt.show()
#too many values and i cant unpack it
#maybe i'll try another solution for this stuff
#so....how 'bout RGB to LAB ??? maybe it'll be possible ?? =)))
from PIL import Image
im = Image.open('F:/coding/Project/FDD/neo.jpg')
im.save('F:/coding/Project/FDD/neo1.png') #nhớ cái chuyển
#convert png into fucking grayscale
from PIL import Image
im = Image.open("F:/coding/Project/FDD/neo1.png")
im_L = im.convert("L")
im_L.save("F:/coding/Project/FDD/neo2.png")
#convert greyscale into RGB 
from PIL import Image
from collections import defaultdict
import pprint

img = Image.open("F:/coding/Project/FDD/neo2.png")
rgbimg = Image.new("RGBA", img.size)
rgbimg.paste(img)

found_colors = defaultdict(int)
for x in range(0, rgbimg.size[0]):
    for y in range(0, rgbimg.size[1]):
        pix_val = rgbimg.getpixel((x, y))
        found_colors[pix_val] += 1 

rgbimg.save('F:/coding/Project/FDD/neo3.png')
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
from past.builtins import xrange
img_file = Image.open("F:/coding/Project/FDD/neo3.png")
img = img_file.load()
[xs, ys] = img_file.size
max_intensity = 100
hues = {}
for x in xrange(0, xs):
  for y in xrange(0, ys):
    [r, g, b] = img[x, y]
    r /= 255.0
    g /= 255.0
    b /= 255.0
    [h, s, v] = colorsys.rgb_to_hsv(r, g, b)
    if h not in hues:
      hues[h] = {}
    if v not in hues[h]:
      hues[h][v] = 1
    else:
      if hues[h][v] < max_intensity:
        hues[h][v] += 1

h_ = []
v_ = []
i = []
colours = []
for h in hues:
  for v in hues[h]:
    h_.append(h)
    v_.append(v)
    i.append(hues[h][v])
    [r, g, b] = colorsys.hsv_to_rgb(h, 1, v)
    colours.append([r, g, b])

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(h_, v_, i, s=5, c=colours, lw=0)
ax.set_xlabel('Hue')
ax.set_ylabel('Value')
ax.set_zlabel('Intensity')
fig.add_axes(ax)
plt.show()
#It's still too many value that i cant unpack it !
