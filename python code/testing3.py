from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

# Load and format data
with cbook.get_sample_data('jacksboro_fault_dem.npz') as file, \
    np.load(file) as dem:
    z = dem['elevation']
    nrows, ncols = z.shape
    x = np.linspace(dem['xmin'], dem['xmax'], ncols)
    y = np.linspace(dem['ymin'], dem['ymax'], nrows)
    x, y = np.meshgrid(x, y)

region = np.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()
-----------------
#test 26/12/2020
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
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
from past.builtins import xrange
img_file = Image.open("F:/coding/Project/FDD/neo.jpg")
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