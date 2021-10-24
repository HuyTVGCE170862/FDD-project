#data heatmap testing
import matplotlib.pyplot as plt
import numpy as np

class ReOrder():
    def __init__(self, array, nrand=2, niter=800):
        self.a = array
        self.indi = np.arange(self.a.shape[0])
        self.indj = np.arange(self.a.shape[1])
        self.i = np.arange(self.a.shape[0])
        self.j = np.arange(self.a.shape[1])
        self.nrand = nrand
        self.niter = niter

    def apply(self, a, i, j):
        return a[:,j][i,:]

    def get_opt(self):
        return self.apply(self.a, self.i, self.j)

    def get_labels(self, x=None, y=None):
        if x is None:
            x = self.indj
        if y is None:
            y = self.indi
        return np.array(x)[self.j], np.array(y)[self.i]

    def cost(self, a=None):
        if a is None:
            a = self.get_opt()
        m = a[1:-1, 1:-1]
        b = 0.5 * ((m - a[0:-2, 0:-2])**2 + \
                   (m - a[2:  , 2:  ])**2 + \
                   (m - a[0:-2, 2:  ])**2 + \
                   (m - a[2:  , 0:-2])**2) + \
            (m - a[0:-2, 1:-1])**2 + \
            (m - a[1:-1, 0:-2])**2 + \
            (m - a[2:  , 1:-1])**2 + \
            (m - a[1:-1, 2:  ])**2 
        return b.sum()

    def randomize(self):
        newj = np.random.permutation(self.a.shape[1])
        newi = np.random.permutation(self.a.shape[0])
        return newi, newj

    def compare(self, i1, j1, i2, j2, a=None):
        if a is None:
            a = self.a
        if self.cost(self.apply(a,i1,j1)) < self.cost(self.apply(a,i2,j2)):
            return i1, j1
        else:
            return i2, j2

    def rowswap(self, i, j):
        rows = np.random.choice(self.indi, replace=False, size=2)
        ir = np.copy(i)
        ir[rows] = ir[rows[::-1]]
        return ir, j

    def colswap(self, i, j):
        cols = np.random.choice(self.indj, replace=False, size=2)
        jr = np.copy(j)
        jr[cols] = jr[cols[::-1]]
        return i, jr

    def swap(self, i, j):
        ic, jc = self.rowswap(i,j)
        ir, jr = self.colswap(i,j)
        io, jo = self.compare(ic,jc, ir,jr)
        return self.compare(i,j, io,jo)

    def optimize(self, nrand=None, niter=None):
        nrand = nrand or self.nrand
        niter = niter or self.niter
        i,j = self.i, self.j
        for kk in range(niter):
            i,j = self.swap(i,j)
        self.i, self.j = self.compare(i,j, self.i, self.j)
        print(self.cost())
        for ii in range(nrand):
            i,j = self.randomize()
            for kk in range(niter):
                i,j = self.swap(i,j)
            self.i, self.j = self.compare(i,j, self.i, self.j)
            print(self.cost())
        print("finished")

#let make it into two arrays

def get_sample_ord():
    	x,y = np.meshgrid(np.arange(12), np.arange(10))
	z = x+y
	j = np.random.permutation(12)
	i = np.random.permutation(10)
	return z[:,j][i,:]

def get_sample():
	return np.random.randint(0,120,size=(10,12))

#and run both of 'em...maybe ! i dont know
#what the fuck am i doing here ?? :v
#whatever :v if it run then dont touch it ! :v

def reorder_plot(nrand=4, niter=10000):
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, 
                                               constrained_layout=True)
    fig.suptitle("nrand={}, niter={}".format(nrand, niter))

    z1 = get_sample()
    r1 = ReOrder(z1)
    r1.optimize(nrand=nrand, niter=niter)
    ax1.imshow(z1)
    ax3.imshow(r1.get_opt())
    xl, yl = r1.get_labels()
    ax1.set(xticks = np.arange(z1.shape[1]),
            yticks = np.arange(z1.shape[0]),
            title=f"Start, cost={r1.cost(z1)}")
    ax3.set(xticks = np.arange(z1.shape[1]), xticklabels=xl, 
            yticks = np.arange(z1.shape[0]), yticklabels=yl, 
            title=f"Optimized, cost={r1.cost()}")

    z2 = get_sample_ord()   
    r2 = ReOrder(z2)
    r2.optimize(nrand=nrand, niter=niter)
    ax2.imshow(z2)
    ax4.imshow(r2.get_opt())
    xl, yl = r2.get_labels()
    ax2.set(xticks = np.arange(z2.shape[1]),
            yticks = np.arange(z2.shape[0]),
            title=f"Start, cost={r2.cost(z2)}")
    ax4.set(xticks = np.arange(z2.shape[1]), xticklabels=xl, 
            yticks = np.arange(z2.shape[0]), yticklabels=yl, 
            title=f"Optimized, cost={r2.cost()}")


reorder_plot(nrand=4, niter=10000)

plt.show()
#as we can see ! it run pretty gud =)))
#now we can change that into pixel of image with using PIL =))

#i turn this into 8bit ! but it look not oke so maybe i'll test it into 16 
from pyxelate import Pyxelate
from skimage import io
import matplotlib.pyplot as plt

img = io.imread("F:/coding/Project/FDD/neo.jpg")
# generate pixel art that is 1/14 the size
height, width, _ = img.shape 
factor = 14
colors = 6
dither = True

p = Pyxelate(height // factor, width // factor, colors, dither)
img_small = p.convert(img)  # convert an image with these settings

_, axes = plt.subplots(1, 2, figsize=(16, 16))
axes[0].imshow(img)
axes[1].imshow(img_small)
plt.show()
#i'll turn this into 16bit
from PIL import Image

def pixelate(input_file_path, pixel_size):
    image = Image.open('F:/coding/Project/FDD/neo.jpg')
    image = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        Image.NEAREST
    )
    image = image.resize(
        (image.size[0] * pixel_size, image.size[1] * pixel_size),
        Image.NEAREST
    )
    image.show()

pixelate("F:/coding/Project/FDD/neo.jpg",16)
#Example: pixelate --input=img/bps.jpg --output=img/bps.png --pixel-size=10