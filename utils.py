import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
import scipy.stats as st
from sklearn.decomposition import PCA
import math
from pathlib import Path

import os
import re
from os.path import join

from descartes import PolygonPatch
import alphashape

import warnings
from shapely.errors import ShapelyDeprecationWarning

import functools
import time
import random

plt.ioff()

##################### Utils ##################################

def invertBinaryImage(im):
    ## Flips the image pixel values(0 and 255) and returns the inverted image
    inv = np.zeros(im.shape)
    inv[im==0] = 255
    return inv

def ConcatAndShow(input, output, scale_percent,text):
    border = np.zeros((input.shape[0],75)) # Setting border between the concatenated images
    img=np.concatenate((input,border,output),axis=1)
    print(text + " \n")
    show_scalled_img(img,scale_percent)

def show_scalled_img(img_arr, scalePercent=100):
    ## shows the image at a scale percent of the origianl size.   
    scale_percent = scalePercent

    #calculate the 50 percent of original dimensions
    width = int(img_arr.shape[1] * scale_percent / 100)
    height = int(img_arr.shape[0] * scale_percent / 100)

    # resize image
    output = cv2.resize(img_arr, (width, height))

    fig = plt.figure(figsize = (50,50))
    plt.imshow(output, cmap='gray', vmin=0, vmax=255)
    plt.show() 

def imgLength(freqCoord, imgSize):
    freqCoord = freqCoord.astype(np.int32)
    beta = np.arctan(imgSize[0]/imgSize[1])

    if freqCoord[0] == 0:
        return imgSize[1]
    elif freqCoord[1] == 0:
        return imgSize[0]
    elif freqCoord[0] > 0 and freqCoord[1] > 0:
        theta = np.pi/2 - np.arctan(freqCoord[1]/freqCoord[0])
    elif freqCoord[0] < 0 and freqCoord[1] < 0:
        theta = np.pi/2 - np.arctan(freqCoord[1]/freqCoord[0])
    elif freqCoord[0] > 0 and freqCoord[1] < 0:
        theta = np.pi/2 + np.arctan(freqCoord[1]/freqCoord[0])
    elif freqCoord[0] < 0 and freqCoord[1] > 0:
        theta = np.pi/2 + np.arctan(freqCoord[1]/freqCoord[0])
    
    if theta <= beta:
        return imgSize[1]/np.cos(theta)
    else:
        return imgSize[0]/np.sin(theta)

def plotHist(value=None, wght=None, path=None, filename=None, numBins=100, logscaling=None, xLabel = None, yLabel=None, show='no'):

    _ = plt.hist(value, weights = wght, bins = numBins)
    # plt.title(filename[:-4])
    
    if logscaling == 'x':
        plt.xscale('log')
    elif logscaling == 'y':
        plt.yscale('log')
    elif logscaling == 'both':
        plt.xscale('log')
        plt.yscale('log')
    
    if xLabel != None:
        plt.xlabel(xLabel)
    if yLabel != None:
        plt.ylabel(yLabel)
    
    plt.savefig(os.path.join(path, filename))
    if show=='yes':
        plt.show()
    plt.close()

def plotScatter(value=None, wght=None, path=None, filename=None, logscaling=None, xLabel = None, yLabel=None, show='no'):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(value, wght, 'o', color = 'black')
    # plt.title(filename[:-4])
    
    lowerLim = min([min(value), min(wght)])
    higherLim = max([max(value), max(wght)])
    
    plt.ylim((lowerLim, higherLim))
    plt.xlim((lowerLim, higherLim))
    ax.set_aspect('equal', adjustable='box')

    if logscaling == 'x':
        plt.xscale('log')
    elif logscaling == 'y':
        plt.yscale('log')
    elif logscaling == 'both':
        plt.xscale('log')
        plt.yscale('log')
    
    if xLabel != None:
        plt.xlabel(xLabel)
    if yLabel != None:
        plt.ylabel(yLabel)
    
    plt.savefig(os.path.join(path, filename))
    
    if show=='yes':
        plt.show()
    plt.close()

def plotKDE(value=None, wght=None, path=None, filename=None, kernel = 'gaussian', bandwidth = None, logscaling=None, xLabel = None, yLabel=None, show='no'):

    minVal  = min(value)
    maxVal  = max(value)
    diff    = int(maxVal - minVal)
    if diff == 0:
        diff = 1
    x_d     = np.linspace(minVal, maxVal, 100*diff)
    # print("minVal   :", minVal)
    # print("maxVal   :", maxVal)
    # print("diff     :", diff)
    # print("x_d.shape:", x_d.shape)
      
    if wght == None:
        value = np.array(value)
        X = value[:, None]

    """ print( "Shape of value before:", value.shape)
    value = np.array(value)[:, None]
    print( "Shape of value:",value.shape)
    wght  = np.array(wght)[:, None]
    X = np.concatenate((value, wght), axis = 1)
    print( "Shape of X:",X.shape) """
    
    if bandwidth == None:
        bandwidth_space  = 10 ** np.linspace(-1, 1, 100)
        grid        = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidth_space}, cv=KFold(5))
        grid.fit(X)
        bandwidth = grid.best_params_['bandwidth']
        print( "Bandwidth Value : ", bandwidth)

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
    log_dens = kde.score_samples(x_d[:,None])

    plt.fill_between(x_d, np.exp(log_dens), alpha=0.5)
    plt.plot(X, np.full_like(X, -0.001), '|k', markeredgewidth=1)

    if logscaling == 'x':
        plt.xscale('log')
    elif logscaling == 'y':
        plt.yscale('log')
    elif logscaling == 'both':
        plt.xscale('log')
        plt.yscale('log')

    if xLabel != None:
        plt.xlabel(xLabel)
    if yLabel != None:
        plt.ylabel(yLabel)
    
    plt.savefig(os.path.join(path, filename))
    if show=='yes':
        plt.show()
    plt.close()

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

def isAreaSmall(area, d_space, factor):
    """
    Determines if the given area is smaller than the threshold area.

    Parameters:
    - area (float): The area to compare.
    - d_space (float): The d-space value (must be in the same units as area).
    - factor (float): The threshold area factor.

    Returns:
    - bool: True if the area is smaller than the threshold; False otherwise.
    """
    width = factor * d_space
    height = d_space
    area_thresh = width * height
    return area < area_thresh

def totalAreaInDSRange(dataframe, dsRange):
    TotDspaceArea   = 0
    dsRangeCount    = 0
    for ind, row in dataframe.iterrows():
        if dsRange[0] <= row['D-Spacing(FFT, nm)'] <= dsRange[1]:
            dsRangeCount += 1
            TotDspaceArea += row['Crystal Area (nm^2)']
    return TotDspaceArea, dsRangeCount

def getCentroid (convHull):
    """ 
    Returns the detection region centroid coordinates according to the image coordinate convention {(columns, rows) with top left as origin}. 
    """
    x_centroid = np.mean(convHull.points[convHull.vertices,0], dtype= int)
    y_centroid = np.mean(convHull.points[convHull.vertices,1], dtype= int)
    return x_centroid, y_centroid

def getCrystalSizeAndOrientation(convHull):
    """ 
    Return's half of the major and minor axis length and the orientation angle of the major axis. 
    The orientation angle is zero at the x-axis and positive in the clockwise direction. 
    """

    X = convHull.points[convHull.vertices,:2]
    pca = PCA(n_components = 2)
    pca.fit(X)

    eigen_vectors = pca.components_
    eigen_values = pca.explained_variance_

    major_scale = np.sqrt(2 * eigen_values[0])
    minor_scale = np.sqrt(2 * eigen_values[1])

    major_axis_unit = eigen_vectors[0, :]
    # minor_axis_unit = eigen_vectors[1, :]

    major_axis = major_axis_unit * major_scale
    # minor_axis = minor_axis_unit * minor_scale
    if major_axis[0] == 0:
        angle = -90
    else:
        angle   = math.atan(major_axis[1]/major_axis[0]) * 180/np.pi 

    return major_scale, minor_scale, angle

def getEquivalentAngle(ang):
    ## Return's angle in [-180, 0]
    if ang > 0:
        ang = -(180-ang)
    return ang

def getAngleDifference(ang1, ang2):
    ang1 = getEquivalentAngle(ang1)
    ang2 = getEquivalentAngle(ang2)
    diff = abs(ang1-ang2) 
    if diff > 90:
        return 180 - diff
    else: 
        return diff

def getBoundingBox(convHull):
    """
    Returns the detection region bounding box coordinates according to the image coordinate convention {(columns, rows) with top left as origin}.  
    x_minMax and y_minMax are list of size 2, containing the min value first and max second. 
    """
    x_minMax    = [np.amin(convHull.points[convHull.vertices,0]), np.amax(convHull.points[convHull.vertices,0])]
    y_minMax    = [np.amin(convHull.points[convHull.vertices,1]), np.amax(convHull.points[convHull.vertices,1])]
    return x_minMax, y_minMax

def pltOrientationLine(subplot, crystalAngle, color, bb_x_minMax, bb_y_minMax, x_centroid, y_centroid):
    arrLen  = min( int( bb_x_minMax[1] - bb_x_minMax[0] ), int( bb_y_minMax[1] - bb_y_minMax[0] ) ) / 6
    subplot.arrow(x_centroid, 
                  y_centroid , 
                  arrLen * np.cos( crystalAngle * np.pi/180 ),
                  arrLen * np.sin( crystalAngle * np.pi/180 ), 
                  linewidth = 7.0, 
                  color = color)

def pltConvexHull(subplot, convHull, pntCloud, color):
    for simplex in convHull.simplices:
            subplot.plot( pntCloud[ simplex , 0 ] , pntCloud[ simplex , 1 ] , linewidth = 7.0 , color = color )

def getAlphaShape(pntCloud, alpha_shape_factor):
    alpha_shape     = alphashape.alphashape( pntCloud , alpha = alpha_shape_factor )
    return alpha_shape


def pltAlphaShape(subplot, alpha_shape):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        subplot.add_patch( PolygonPatch(alpha_shape, alpha = 0.2) )

def createVersionDirectory(folderDir, name):
    
    # Check folderDir exists else create it
    if not folderDir.exists():
        folderDir.mkdir(parents=True, exist_ok=True)
    
    # folderDir = Path(projectPath) / BaseResultDir
    ResFolderName = name + '_'  # 'version_'
    lenFolderName = len(ResFolderName)

    dirlist = [int(item.name[lenFolderName:]) 
               for item in folderDir.iterdir() 
               if (item.is_dir() 
                   and re.search(ResFolderName, item.name) 
                   and len(item.name) > lenFolderName
                   )
               ]

    latestVersion = max(dirlist, default=0)
    new_version_dir = folderDir / f"{ResFolderName}{latestVersion + 1}"
    new_version_dir.mkdir()

    print(f"{ResFolderName}{latestVersion + 1}\n")
    return new_version_dir
    
# def filterThreshArea(df, params):
def filterThreshArea(df, 
                     df_area_col_name, 
                     d_space, 
                     threshold_area_factor):
    # area_col_name       = 'Crystal Area (nm^2)'
    # threshold_area_factor = params['threshold area factor']
    
    # Check if column names are present in the dataframe
    if df_area_col_name not in df.columns:
        raise ValueError(f"Column '{df_area_col_name}' not found in dataframe")
    
    for ind, row in df.iterrows():
        # area and d_space both in same units (either nm or pix)
        area = row[df_area_col_name]
        TorF_area = isAreaSmall(area, d_space, threshold_area_factor)
        if TorF_area == True:# or row['D-Spacing(FFT, nm)']<1.5:
            df.drop(ind, inplace = True)
    
    return df        

def CreateDirectories(directories):
    for directory in directories:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            
            
def calculate_pixel_size(value, factor):
    return int(value * factor)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if func.__name__ == 'process_images' or func.__name__ == 'process_images_serial' or func.__name__ == 'main':
            print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def minDist(pts1, pts2):
    # Compute all pairwise distances
    dists = np.linalg.norm(pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :], axis=2)

    # Return the minimum distance
    return np.min(dists)

def create_rgb_image(img):
    return cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2RGB)

def load_img_result_dir(img_path, params):
    BGRImg = cv2.imread(join(params['result image directory'], img_path.stem + params['save image format']))
    RGBImg = cv2.cvtColor(BGRImg, cv2.COLOR_BGR2RGB)
    return RGBImg

def pick_unique_colors(count):
    
    color_options = ['b', 'r', 'c', 'm','y','w']
    # color_options = ['b', 'g', 'r', 'c', 'm','y','w']
    
    assert count <= len(color_options), "Count should be less than or equal to the number of colors available"
    crystal_color = random.sample(color_options, count)
    # print("Crystal color:", crystal_color)
    
    return crystal_color
