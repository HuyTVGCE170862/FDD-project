import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# load image
img = mpimg.imread('../../doc/_static/stinkbug.png')
print(img)
imgplot = plt.imshow(img)
# channel of our data:
lum_img = img[:, :, 0]
plt.imshow(lum_img)
#hot
plt.imshow(lum_img, cmap="hot")
#spectral
imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')
#colorbar
imgplot = plt.imshow(lum_img)
plt.colorbar()
# Data ranges
# Examining a specific data range
# ---------------------------------
plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# specify the clim
imgplot = plt.imshow(lum_img, clim=(0.0, 0.7))
###############################################################################
# You can also specify the clim using the returned object
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(lum_img)
ax.set_title('Before')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.0, 0.7)
ax.set_title('After')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
# .. _Interpolation:
# Array Interpolation schemes
from PIL import Image
img = Image.open('../../doc/_static/stinkbug.png')
img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place
imgplot = plt.imshow(img)
# Let's try some others. Here's "nearest", which does no interpolation.
imgplot = plt.imshow(img, interpolation="nearest")
# and bicubic:
imgplot = plt.imshow(img, interpolation="bicubic")
