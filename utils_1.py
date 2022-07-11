import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
import scipy.stats as st
from sklearn.decomposition import PCA
import math

import os
import re
from os.path import join

from descartes import PolygonPatch
import alphashape

def plotKDE_2D(value=None, wght=None, path=None, filename=None, kernel = 'gaussian', bandwidth = None, logscaling=None, xLabel = None, yLabel=None, show='no'):
    
    # Extract x and y# Extract x and y
    x = np.array(value)
    y = np.array(wght)

    # Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10

    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

    if xLabel != None:
        ax.set_xlabel(xLabel)
    if yLabel != None:
        ax.set_ylabel(yLabel)

    if logscaling == 'x':
        ax.set_xscale('log')
    elif logscaling == 'y':
        ax.set_yscale('log')
    elif logscaling == 'both':
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
    ax.view_init(45, 0)
    
    fig.savefig(os.path.join(path, filename))
    if show=='yes':
        plt.show()
    plt.close()

    h =plt.hist2d(x, y, bins=(36, 36))
    plt.colorbar(h[3])
    if xLabel != None:
        plt.xlabel(xLabel)
    if yLabel != None:
        plt.ylabel(yLabel)
    plt.savefig(os.path.join(path, '2D_hist_'+filename))
    if show=='yes':
        plt.show()
    plt.close()

def createVersionDirectory(projectPath, BaseResultDir, name):
    folderDir       = join(projectPath, BaseResultDir)
    ResFolderName   = name + '_' #'version_'
    lenFolderName   = len(ResFolderName)
    dirlist         = [int(item[lenFolderName:]) for item in os.listdir(folderDir) if os.path.isdir(os.path.join(folderDir,item)) and re.search(ResFolderName, item) != None and len(item)> lenFolderName] 
    
    if len(dirlist) != 0:
        latestVersion = max(dirlist)
    else:
        latestVersion = 0

    os.mkdir(join(folderDir, ResFolderName + str(latestVersion + 1)))
    print(ResFolderName + str(latestVersion + 1) + '\n')
    resultDir = join(BaseResultDir,ResFolderName + str(latestVersion + 1))
    return resultDir