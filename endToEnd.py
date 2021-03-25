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

    fig = plt.figure(figsize = (50,50))
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


############### OPERATIONS USED IN GRATE ############################3

def BlurThresh(img, num_interation):
    input = img
    thresh = input
    for i in range(num_interation):
        blur = cv2.GaussianBlur(thresh,(15,15),0)
        _,thresh = cv2.threshold(blur,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

# Closing: Removing black spots from white regions. Basically Dilation followed by Erosion.
def Closing(img, k_size):
    input = img
    kernel = np.ones((k_size,k_size),np.uint8)
    output = cv2.morphologyEx(input, cv2.MORPH_CLOSE, kernel)
    return output

## Opening: Removing white spots from black regions. Basically Erosion followed by Dilation.
def Opening(img, k_size):
    input = img
    kernel = np.ones((k_size,k_size),np.uint8)
    output = cv2.morphologyEx(input, cv2.MORPH_OPEN, kernel)
    return output 

## Skeletonize Image
def Skeletonize(img): 
    input = img 
    image = invert(input/np.max(input))
    skeleton = skeletonize(image)
    output = (skeleton/np.max(skeleton))*255 
    return output

## Finds branching points for images with black background and white lines receive a degree matrix
def BreakBraches(img):
    input = np.copy(img)
    _, _, degrees = skeleton_to_csgraph(input)
    # consider all values larger than two as intersection
    intersection_matrix = degrees > 2
    input[intersection_matrix==True] = 0
    return input

## Skeleton Segmentation
def SkeletonSegmentation(img):
    input = np.copy(img)
    input = input.astype('uint8')
    # Set global debug behavior to None (default), "print" (to file), or "plot" (Jupyter Notebooks or X11)
    pcv.params.debug = "None"
    pcv.params.line_thickness = 2
    segmentedImg, obj = pcv.morphology.segment_skeleton(skel_img=input)
    
    return segmentedImg, obj

## Filtering out small backbones using the pixThresh variable and breaking them into uniform size.
def Filtered_Uniform_BB(img, obj_list, pixelThreshold, ellipseSize):
    bb = np.zeros(img.shape)
    for i in range(len(obj_list)):
        if len(obj_list[i]) > pixelThreshold:
            count = 0
            for ind in obj_list[i]:
                count += 1
                if count%ellipseSize == 0:
                    bb[ind[0][1], ind[0][0]] = 0
                else:
                    bb[ind[0][1], ind[0][0]] = 255
    return bb

## Removing unnecessary dimension from the filteredPoly and storing it in restructuredFP
def RemovingDim(lis):
    temp = []
    for i in range(len(lis)):
        tp = np.zeros((len(lis[i]), 2))
        for j,val in enumerate(lis[i]):
            tp[j][0] = val[0][0]
            tp[j][1] = val[0][1]
        temp.append(tp)
    return temp


################ SCIPY BASED ELLIPSE CONSTRUCTION FUNCTION ###############################
def EllipseConstruction(img, aspectRatio):
    input = np.copy(img)
    label_img = label(input)
    print("MODIFIED Num Unique Values in IMG:", len(np.unique(label_img)))
    props = regionprops_table(label_img, properties=('centroid',
                                                     'orientation',
                                                     'major_axis_length',
                                                     'minor_axis_length'))
    props = pd.DataFrame(props)
    props['orientation'] = 90 - props['orientation']*180/np.pi
    props['major_axis_length'] = props['major_axis_length']/2
    props['minor_axis_length'] = props['minor_axis_length']/2

    bb_props = pd.DataFrame()
    bb_props = props
    
    for ind in range(props.shape[0]):
        if props['minor_axis_length'][ind] < 1:
            bb_props = bb_props.drop([ind])   
        elif props['major_axis_length'][ind]/props['minor_axis_length'][ind] > aspectRatio:
            ellip_temp_img = cv2.ellipse(input, (int(props['centroid-1'][ind]), int(props['centroid-0'][ind])), (int(props['major_axis_length'][ind]),int(props['minor_axis_length'][ind])),props['orientation'][ind], 0.0, 360.0, (255, 0, 0), 2);
        else: 
            bb_props = bb_props.drop([ind])
    return ellip_temp_img, bb_props

## Creating Adjacency Matrix based on distance between centroid and the orientation angle
def AdjacencyMat(img, bb_props, distanceThresh, thetaThresh):
    
    bb_props_np = bb_props.to_numpy()
    N = len(bb_props_np)
    A_Mat = np.zeros((N,N))
    print("adjacencyMat Size: ",N)
    for i in range(N):
        for j in range(N): ## Change j to i+1 to N
            if i == j:
                continue
            else:
                l2norm = np.linalg.norm([bb_props_np[i][0]-bb_props_np[j][0],bb_props_np[i][1]-bb_props_np[j][1]])
                angleDiff = abs(bb_props_np[i][2]-bb_props_np[j][2])
                if l2norm < distanceThresh and angleDiff < thetaThresh:
                    A_Mat[i][j] = 1
                    A_Mat[j][i] = 1
    
    # Plotting the adjacent ellipse.
    A_Mat_img = np.copy(img)
    for i in range(N):
        for j in range(N):
            if A_Mat[i][j] == 1:
                poly = bb_props_np[i]
                A_Mat_img = cv2.ellipse(A_Mat_img, (int(poly[1]), int(poly[0])), (int(poly[3]), int(poly[4])), poly[2], 0.0, 360.0, (255, 0, 0), 2);
                break
                
    return A_Mat_img, A_Mat

## Connected Components:
def ConnecComp(img, A_Mat, props, c_size):
    polyProps = props.to_numpy() 
    N = A_Mat.shape[0]
    visited = np.zeros(N)
    cc = []
    
    for v in range(N):
        if visited[v] == 0:
            temp = []
            cc.append(DFSUtil(temp, v, visited, N, A_Mat))
    
    ccImg = np.copy(img)
    print("Length of CC: ",len(cc))
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0)
    thickness = 2

    AllClusterPointCloud = []
    crystalAngles = []
    print("Cluster Size:", c_size)
    for i in range(len(cc)):
        if len(cc[i]) >= c_size:
            print("cc ind:", i)
            majorsAxisPointCloud = []
            ellipseAngels = []
            for j in cc[i]:
                poly = polyProps[j]
                ccImg = cv2.ellipse(ccImg, (int(poly[1]), int(poly[0])), (int(poly[3]), int(poly[4])), poly[2], 0.0, 360.0, (255, 0, 0), 2);
                temp = np.zeros([2,2])
                ang = poly[2]*np.pi/180
                temp[0,0] = int(poly[1] - poly[3]*np.cos(ang)) 
                temp[0,1] = int(poly[0] - poly[3]*np.sin(ang))

                temp[1,0] = int(poly[1] + poly[3]*np.cos(ang))
                temp[1,1] = int(poly[0] + poly[3]*np.sin(ang))
                #ccImg = cv2.circle(ccImg, (int(temp[0,0]),int(temp[0,1])), radius=5, color=(255, 0, 0), thickness=-1)
                #ccImg = cv2.circle(ccImg, (int(temp[1,0]),int(temp[1,1])), radius=5, color=(255, 0, 0), thickness=-1)
                majorsAxisPointCloud.append(temp[0,:])
                majorsAxisPointCloud.append(temp[1,:])
                ellipseAngels.append(poly[2])
            AllClusterPointCloud.append(majorsAxisPointCloud)
            crystalAngles.append(mean(ellipseAngels))
    
    return ccImg, AllClusterPointCloud, crystalAngles

def PlottingAndSaving(img, ClusterPointCloud, ProjectPath, ResultDir, ImgName):
    RGBImg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    CrystalImg = img#invertBinaryImage(finalImg)

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize = (50,25))
    axes[1].imshow(RGBImg, vmin=0, vmax=255)

    crystalArea = []
    for cluster in ClusterPointCloud:

        PointCloud = np.array(cluster)
        hull = ConvexHull(PointCloud)
        crystalArea.append(hull.area)
        color = ['b', 'g', 'r', 'c', 'm','y','w']
        c= random.choice(color)
        for simplex in hull.simplices:
            plt.plot(PointCloud[simplex, 0], PointCloud[simplex, 1],linewidth=7.0, color=c)

    axes[0].imshow(img, cmap = 'gray')
    figure.tight_layout()
    figure.savefig(ProjectPath + ResultDir + ImgName[:-4]+'.png')
    plt.show()
    figure.clf()
    plt.close()
    plt.clf()
    gc.collect()
    return crystalArea

def ConcatAndShow(input, border, output, scale_percent,text):
    img=np.concatenate((input,border,output),axis=1)
    print(text + " \n")
    show_scalled_img(img,scale_percent)

## GRATE as Function

def GRATE(projectPath, dataDir, imgName, resultDir):
    img = cv2.imread(projectPath + dataDir + imgName,0)
    
    image_scale_percent = 50 # Scaling the image before display
    border = np.zeros((img.shape[0],75)) # Setting border between the concatenated images
    blur_thresh_interation = 40
    closing_k_size = 15 # Kernel Size
    opening_k_size = 17 # Kernel Size
    pixThresh = 150 # Threshold number of pixels consituting polymers
    ellipse_len = 160 # Breaking polymer into this size before constructing ellipse 
    ellipseAspectRatio = 5 # Threshold aspect Ratio of the ellipse
    thresh_dist = 200  # Centroid Distance threshold for adjacency matrix 
    thresh_theta = 25  # delta Theta threshold for adjacency matrix 
    clusterSize = 10 # Threshold Crystal cluster size 
    

    thresh = BlurThresh(img, blur_thresh_interation)
#     ConcatAndShow(img, border, thresh, image_scale_percent, "BLURRING AND THRESHOLDING")
    
    closing = Closing(thresh, closing_k_size)
#     ConcatAndShow(thresh, border, closing, image_scale_percent, "CLOSING")
    
    opening = Opening(closing, opening_k_size)
#     ConcatAndShow(closing, border, opening, image_scale_percent, "OPENING")
    
    skeleton = Skeletonize(opening)
#     ConcatAndShow(opening, border, skeleton, image_scale_percent, "SKELETONIZED")
    
    skeleton = BreakBraches(skeleton)
#     print("BRANCHED SKELETON \n")
#     show_scalled_img(skeleton, image_scale_percent)
    
    _, temp = SkeletonSegmentation(skeleton)
    
    Broken_backbone_img = Filtered_Uniform_BB(img, temp, pixThresh, ellipse_len)
#     print("UNIFORM SIZED BACKBONE \n")
#     show_scalled_img(Broken_backbone_img, image_scale_percent)
    
    _, filteredPolys = SkeletonSegmentation(Broken_backbone_img)

    restructuredFP = RemovingDim(filteredPolys)
    
    bb_ellipse1, bb_ellipse_props = EllipseConstruction(Broken_backbone_img, ellipseAspectRatio)
#     print("ELLIPSE INSCRIBED \n")
#     show_scalled_img(bb_ellipse1,image_scale_percent)
    
    adjEllipse_img, adjacencyMat = AdjacencyMat(Broken_backbone_img, bb_ellipse_props, thresh_dist, thresh_theta)
#     print("ADJACENT ELLIPSES \n")
#     show_scalled_img(adjEllipse_img, image_scale_percent)
    
    ellipseCluster, AllClusterPointCloud, crystalAngles = ConnecComp(Broken_backbone_img, adjacencyMat, bb_ellipse_props, clusterSize)
#     print("CLUSTERS")
#     show_scalled_img(ellipseCluster, image_scale_percent)
    
    crystalArea = PlottingAndSaving(img, AllClusterPointCloud, projectPath, resultDir, imgName)
        
    df = pd.DataFrame(list(zip(crystalArea, crystalAngles)), columns=['Crystal Area', 'Crystal Angle'])
    df.to_csv(projectPath + resultDir +imgName[:-4]+'.csv')
    
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