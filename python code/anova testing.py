import numpy as np
import matplotlib.pyplot as plt
import pandas
# For 3d plots. This import is necessary to have 3D plotting below
from mpl_toolkits.mplot3d import Axes3D
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
##############################################################################
# Generate and show the data
x = np.linspace(-28, 34, 124)
# We generate a 2D grid
X, Y = np.meshgrid(x, x)
# To get reproducable values, provide a seed value
np.random.seed(1)
# Z is the elevation of this 2D grid
Z = -193 + 254*X - 100*Y + 190 * np.random.normal(size=X.shape)
# Plot the data
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                       rstride=1, cstride=1)
ax.view_init(900, -270)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
##############################################################################
# Multilinear regression model, calculating fit, P-values, confidence
# intervals etc.
# Convert the data into a Pandas DataFrame to use the formulas framework
# in statsmodels
# First we need to flatten the data: it's 2D layout is not relevent.
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
data = pandas.DataFrame({'x': X, 'y': Y, 'z': Z})
# Fit the model
model = ols("z ~ x + y", data).fit()
# Print the summary
print(model.summary())
print("\nRetrieving manually the parameter estimates:")
print(model._results.params)
# should be array([-4.99754526,  3.00250049, -0.50514907])
# Peform analysis of variance on fitted linear model
anova_results = anova_lm(model)
print('\nANOVA results')
print(anova_results)
plt.show()
------#-------
import numpy as numpy
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
x, y = make_blobs(n_samples=300, centerss=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0],X[:,1])
plt.show()
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlable('Number of Cluster')
plt.ylable('WCSS')
plt.show()
kmeans = KMeans(n_cluster=4, init='k-means++', max_iter=300, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatterkmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()
from pandas import DataFrame
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }

df = DataFrame(Data,colums=['x', 'y'])
print (df)
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
df = Dataframe(Data,columns=['x', 'y'])
kmeans = KMeans(n_cluster=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(df['x'], df['y'], c= kmeans.lables_.astype(float), s=50, alpha=0,5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
from pandas import DataFrame 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
df = DataFrame(Data, columns=['x', 'y'])
kmeans = KMeans(n_cluster=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(df['x'], df['y'], c= kmeans.lables_.astype(float), s=5-, alpha=0.5)
plt.show()
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
df = DataFrame(Data,columns=['x', 'y'])
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
#------deo code nua ! met roi ! : )
import tkinker as tk
from tkinker import filedialog
import pandas as pd
from pandas import DataFrame 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root= tk.Tk()
canvas1 = tk.Canvas(root, width = 400, heigh = 300, relief = 300)
canvas1.pack

label1 = tk.Lable(root, text='k-Means Clustering')
lable1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=lable1)
lable2 = tk.Lable(root, text='Type Number of Cluster:')
lable2.config(fotn=('helvatica', 8))
canvas1. create_window(200, 120, window=lable2)
entry1= tk.Entry (root)
canvas1.create_window(200, 140, window=entry1)
def getExcel ():
        global df
        import_file_path = filedialog.askopenfilename('C:/file_path_go_here')
        read_file = pd.read_excel ('file path on')
        df = DataFrame(read_file,columns=['x', 'y'])
        browseButtonExcel = tk.Button(text=" Import Excel File")
        canvas1.create_window(200, 70, window=browseButtonExcel)

def getaKmeans ():
        global df
        global numberOFCluster
        numberOFCluster = int(entry1.get())
        kmeans = KMeans(n_clusters=numberOFCluster).fit(df)
        centroids = kmeans.cluster_centers_

        lable3 = tk.Figure(figsize=(4,3), dpi=100)
        ax1.scatter(centroids[:, 0], centroidsp[:, 1], c='red', s=50)
        ax1.scatter(df['x'], df['y'], c= kmeans.lables_.astype(float), s=50, alpha=0,5)
        for i, lab in enumerate(['R','G','B'], 1):
                temp = np.zeros(image_rgb.shape)
                temp[:,:,i - 1] = image_rgb[:,:,i - 1]
        ax[i].imshow(temp/255.0) 
        ax[i].axis("off")
        ax[i].set_title(lab)

plt.show()
#cặc !đéo code nữa ! :) ngủ nghỉ ! :)
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
ax[3].imshow(image_lab[:,:,2], cmap='YlGnBu_r')



