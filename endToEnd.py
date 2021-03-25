#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skan import skeleton_to_csgraph
from skimage import io, morphology
from plantcv import plantcv as pcv
import math
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
from scipy import spatial
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table, EllipseModel
from skimage.transform import rotate
import pandas as pd 
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random
from statistics import mean 
import gc

def invertBinaryImage(im):
    ## Flips the image pixel values(0 and 255) and returns the inverted image
    inv = np.zeros(im.shape)
    inv[im==0] = 255
    return inv

def show_scalled_img(img_arr, scalePercent=100):
    ## shows the image at a scale percent of the origianl size.   
    scale_percent = scalePercent

    #calculate the 50 percent of original dimensions
    width = int(img_arr.shape[1] * scale_percent / 100)
    height = int(img_arr.shape[0] * scale_percent / 100)

    # dsize
    #dsize = (width, height)

    # resize image
    output = cv2.resize(img_arr, (width, height))

    plt.figure(figsize = (50,50))
    plt.imshow(output, cmap='gray', vmin=0, vmax=255)
    plt.show()

## Connected Components: 
def DFSUtil(temp, v, visited, numEllipse, adjacencyMat):
    N = numEllipse
    # Mark the current vertex as visited
    visited[v] = 1
    
    # Store the vertex to list 
    temp.append(v)
    
    # Repeat for all vertices adjacent
    for i in range(N):
        if adjacencyMat[v][i] == 1:
            if visited[i] == 0:
                # Update the list
                temp = DFSUtil(temp, i, visited, N, adjacencyMat)
    #print(temp)
    return temp


## GRATE as Function

def GRATE(projectPath, dataDir, imgName, resultDir):
    img = cv2.imread(projectPath + dataDir + imgName,0)
    if img.size == 0:
        print("FILE NOT PROPERLY LOADED"+ projectPath + dataDir + imgName+" \n")
        return 0
    #image_scale_percent = 50 # Scaling the image before display
    #border = np.zeros((img.shape[0],75)) # Setting border between the concatenated images
    blur_thresh_interation = 20
    closing_k_size = 15 # Kernel Size
    opening_k_size = 17 # Kernel Size
    pixThresh = 200 # Threshold number of pixels consituting polymers
    ellipseAspectRatio = 5 # Threshold aspect Ratio of the ellipse
    thresh_dist = 300  # Centroid Distance threshold for adjacency matrix 
    thresh_theta = 15  # delta Theta threshold for adjacency matrix 
    clusterSize = 7 # Threshold Crystal cluster size  

    ## Blurring and Thresholding 
    input = img
    thresh = input
    for i in range(blur_thresh_interation):
        blur = cv2.GaussianBlur(thresh,(15,15),0)
        _,thresh = cv2.threshold(blur,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Closing: Removing black spots from white regions. Basically Dilation followed by Erosion.
    input = thresh
    kernel = np.ones((closing_k_size,closing_k_size),np.uint8)
    closing = cv2.morphologyEx(input, cv2.MORPH_CLOSE, kernel)

    ## Opening  : Removing white spots from black regions. Basically Erosion followed by Dilation.
    input = closing
    kernel = np.ones((opening_k_size,opening_k_size),np.uint8)
    opening = cv2.morphologyEx(input, cv2.MORPH_OPEN, kernel)

    ## Skeletonize Image after opening 
    input = opening 
    image = invert(input/np.max(input))
    skeleton = skeletonize(image)
    skeleton = (skeleton/np.max(skeleton))*255 

    
    ## Finds branching points for images with black background and white lines
    # receive a degree matrix
    _, _, degrees = skeleton_to_csgraph(skeleton)

    # consider all values larger than two as intersection
    intersection_matrix = degrees > 2
    skeleton[intersection_matrix==True] = 0
    #show_scalled_img(skeleton, image_scale_percent)

    ## Skeleton Segmentation

    # Set global debug behavior to None (default), "print" (to file), 
    # or "plot" (Jupyter Notebooks or X11)
    pcv.params.debug = "None"

    # Adjust line thickness with the global line thickness parameter (default = 5),
    # and provide binary mask of the plant for debugging. NOTE: the objects and
    # hierarchies returned will be exactly the same but the debugging image (segmented_img)
    # will look different.
    pcv.params.line_thickness = 2
    skeleton = skeleton.astype('uint8')
    _, obj = pcv.morphology.segment_skeleton(skel_img=skeleton)

    #show_scalled_img(segmented_img, image_scale_percent)

    ## Filtering out small polymer branches using the pixThresh variable.  
    test = np.zeros(img.shape)
    filteredPolys = []

    for i in range(len(obj)):
        if len(obj[i]) > pixThresh:
            filteredPolys.append(obj[i])
            for ind in obj[i]:
                test[ind[0][1], ind[0][0]] = 255

    #show_scalled_img(test, image_scale_percent)


    ## Removing unnecessary dimension from the filteredPoly and storing it in restructuredFP
    restructuredFP = []
    for i in range(len(filteredPolys)):
        tp = np.zeros((len(filteredPolys[i]), 2))
        for j,val in enumerate(filteredPolys[i]):
            tp[j][0] = val[0][0]
            tp[j][1] = val[0][1]
        restructuredFP.append(tp)

    input = test#np.copy(test)
    label_img = label(input)
    props = regionprops_table(label_img, properties=('centroid',
                                                     'orientation',
                                                     'major_axis_length',
                                                     'minor_axis_length'))
    props = pd.DataFrame(props)
    props['orientation'] = 90 - props['orientation']*180/np.pi
    props['major_axis_length'] = props['major_axis_length']/2
    props['minor_axis_length'] = props['minor_axis_length']/2

    pd_backbone_ellipse_props = pd.DataFrame()
    pd_backbone_ellipse_props = props

    for ind in range(props.shape[0]):
    #     if props['major_axis_length'][ind]/props['minor_axis_length'][ind] > ellipseAspectRatio:
    #         ellip_temp_img = cv2.ellipse(input, (int(props['centroid-1'][ind]), int(props['centroid-0'][ind])), (int(props['major_axis_length'][ind]),int(props['minor_axis_length'][ind])),props['orientation'][ind], 0.0, 360.0, (255, 0, 0), 2);
    #     else: 
        if props['minor_axis_length'][ind] < 1:
            pd_backbone_ellipse_props = pd_backbone_ellipse_props.drop([ind])

        elif props['major_axis_length'][ind]/props['minor_axis_length'][ind] < ellipseAspectRatio:
            pd_backbone_ellipse_props = pd_backbone_ellipse_props.drop([ind])

    PolysWithProps = pd_backbone_ellipse_props.to_numpy()
    #show_scalled_img(ellip_temp_img)

    ## Creating Adjacency Matrix based on distance between centroid and the orientation angle
    N = len(PolysWithProps)
    adjacencyMat = np.zeros((N,N))

    for i in range(N):
        for j in range(N): ## Change j to i+1 to N
            if i == j:
                continue
            else:
                #l2norm = np.linalg.norm(PolysWithProps[i][0]-PolysWithProps[j][0])
                l2norm = np.linalg.norm([PolysWithProps[i][0]-PolysWithProps[j][0],PolysWithProps[i][1]-PolysWithProps[j][1]])
                #angleDiff = abs(PolysWithProps[i][1]-PolysWithProps[j][1])
                angleDiff = abs(PolysWithProps[i][2]-PolysWithProps[j][2])
                if l2norm < thresh_dist and angleDiff<thresh_theta:
                    adjacencyMat[i][j] = 1
                    adjacencyMat[j][i] = 1

    ## Connected Components:
    visited = np.zeros(N)
    cc = []

    for v in range(N):
        if visited[v] == 0:
            #print(v)
            temp = []
            cc.append(DFSUtil(temp, v, visited, N, adjacencyMat))

    # finalImg = np.copy(test)

    # startAngle = 0
    # endAngle = 360
    # color = (255, 0, 0)
    # thickness = 2

    AllClusterPointCloud = []
    crystalAngles = []
    crystalArea = []
    for i in range(len(cc)):
        if len(cc[i]) >= clusterSize:
            majorsAxisPointCloud = []
            ellipseAngels = []
            for j in cc[i]:
                poly = PolysWithProps[j]
                temp = np.zeros([2,2])
                ang = poly[2]*np.pi/180
                temp[0,0] = int(poly[1] - poly[3]*np.cos(ang)) 
                temp[0,1] = int(poly[0] - poly[3]*np.sin(ang))

                temp[1,0] = int(poly[1] + poly[3]*np.cos(ang))
                temp[1,1] = int(poly[0] + poly[3]*np.sin(ang))
                majorsAxisPointCloud.append(temp[0,:])
                majorsAxisPointCloud.append(temp[1,:])
                ellipseAngels.append(poly[2])
            AllClusterPointCloud.append(majorsAxisPointCloud)
            crystalAngles.append(mean(ellipseAngels))


    RGBImg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize = (50,25))
    axes[1].imshow(RGBImg, vmin=0, vmax=255)


    for cluster in AllClusterPointCloud:

        PointCloud = np.array(cluster)
        hull = ConvexHull(PointCloud)
        crystalArea.append(hull.area)
        color = ['b', 'g', 'r', 'c', 'm','y','w']
        c= random.choice(color)
        for simplex in hull.simplices:
            plt.plot(PointCloud[simplex, 0], PointCloud[simplex, 1],linewidth=7.0, color=c)

    axes[0].imshow(img, cmap = 'gray')
    figure.tight_layout()
    figure.savefig(projectPath + resultDir + imgName[:-4]+'.png')
    
    figure.clf()
    plt.close()
    plt.clf()
    #plt.show()
    del RGBImg, img
    gc.collect()
    
    df = pd.DataFrame(list(zip(crystalArea, crystalAngles)), columns=['Crystal Area', 'Crystal Angle'])
    df.to_csv(projectPath + resultDir +imgName[:-4]+'.csv')

    del filteredPolys,restructuredFP, cc, AllClusterPointCloud,crystalAngles,crystalArea

    return 0

from os import listdir
from os.path import isfile, join
import sys

#projectPath = '/home/dhruv/Dhruv/ISU/PhD/GRATE/GRATE_for_PennState/'
#dataDir = 'sampleData/'
projectPath = '/work/baskarg/Dhruv/GRATE/'
dataDir = 'data/' + str(sys.argv[1]) + '/'
resultDir = 'Results/'+ str(sys.argv[1]) + '/'
onlyfiles = [f for f in listdir(projectPath+dataDir) if isfile(join(projectPath+dataDir, f))]

for f in onlyfiles:
    #print(projectPath+dataDir+f)
    GRATE(projectPath, dataDir, f, resultDir)


# ## END