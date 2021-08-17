import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
from os.path import join

from descartes import PolygonPatch
import alphashape

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

def debugORSave(initial, final, params, concat, text):
    if params['show intermediate images'] == 1:
        if concat == 1:
            ConcatAndShow(initial, final, params['display image scaling'], text)
        else:
            print(text + "\n")
            show_scalled_img(final, params['display image scaling'])
    if params['save intermediate images'] == 1:
        cv2.imwrite(text+".png", final)

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

def plotfig(value, path, filename, numBins, wght=None, semilog = 1):
    _ = plt.hist(value, weights = wght,bins=numBins)
    plt.title(filename[:-4])
    if semilog == 1:
        plt.semilogx()
    plt.savefig(os.path.join(path, filename))
    plt.show()

def isAreaSmall(area, params):
    factor          = params['Threshold area factor']
    d_space         = params['d space pix']
    width           = factor*d_space
    height          = d_space
    areaThresh      = width*height

    if area < areaThresh:
        return True
    else: 
        return False

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

def getBoundingBox(convHull):
    """ 
    Returns the detection region bounding box coordinates according to the image coordinate convention {(columns, rows) with top left as origin}.  
    x_minMax and y_minMax are list of size 2, containing the min value first and max second. 
    """
    x_minMax    = [np.amin(convHull.points[convHull.vertices,0]), np.amax(convHull.points[convHull.vertices,0])]
    y_minMax    = [np.amin(convHull.points[convHull.vertices,1]), np.amax(convHull.points[convHull.vertices,1])]
    return x_minMax, y_minMax

def pltOrientationLine(axes, crystalIndex, crystalAngle, color, bb_x_minMax, bb_y_minMax, x_centroid, y_centroid):
    arrLen  = min( int( bb_x_minMax[1] - bb_x_minMax[0] ) , int( bb_y_minMax[1] - bb_y_minMax[0] ) ) / 6
    axes[1].arrow( x_centroid, y_centroid , arrLen * np.cos( crystalAngle[ crystalIndex ] * np.pi/180 ) , arrLen * np.sin( crystalAngle[crystalIndex] * np.pi/180 ) , linewidth = 7.0 , color = color )

def pltConvexHull(axes, convHull, pntCloud, color):
    for simplex in convHull.simplices:
            axes[1].plot( pntCloud[ simplex , 0 ] , pntCloud[ simplex , 1 ] , linewidth = 7.0 , color = color )

def pltAlphaShape(axes, pntCloud, params):
    alpha_shape     = alphashape.alphashape( pntCloud , alpha = 0.005 )

    ## Filtering out detections with very small area, Threshold area = factor*(d_space**2).
    TorF = isAreaSmall(alpha_shape.area, params)
    
    if TorF == False:
        axes[1].add_patch( PolygonPatch( alpha_shape , alpha = 0.2 ) )
        
    return alpha_shape, TorF

def createVersionDirectory(projectPath, BaseResultDir, name):
    folderDir       = join(projectPath, BaseResultDir)
    ResFolderName   = name + '_' #'version_'
    lenFolderName   = len(ResFolderName)
    dirlist         = [int(item[lenFolderName:]) for item in os.listdir(folderDir) if os.path.isdir(os.path.join(folderDir,item)) and re.search(ResFolderName, item) != None and len(item)> lenFolderName] 
    print("printing dirlist:", dirlist)

    if len(dirlist) != 0:
        latestVersion = max(dirlist)
    else:
        latestVersion = 0

    print("max of dirlist:", latestVersion)
    os.mkdir(join(folderDir, ResFolderName + str(latestVersion + 1)))
    resultDir = join(BaseResultDir,ResFolderName + str(latestVersion + 1))
    return resultDir