import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.lib.ufunclike import _fix_out_named_y
# from scipy.spatial.kdtree import KDTree
from sklearn.neighbors import KDTree
from skan import skeleton_to_csgraph
from plantcv import plantcv as pcv
import math
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.measure import label, regionprops, regionprops_table
from scipy.signal.signaltools import wiener
import pandas as pd 
from scipy.spatial import ConvexHull
import random
from statistics import mean 
import gc
from descartes import PolygonPatch
import alphashape
from sklearn.decomposition import PCA
from os.path import join
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from utils import *

############### Operations ############################3

def BlurThresh(img, params):
    blur_img    = img
    k_size      = params['blur k size']
    
    if k_size%2 != 1:
        k_size = k_size+1

    for i in range( params[ 'blur iterations' ] ):
        blur_img = cv2.blur( blur_img , ( k_size , k_size ) )

    blur_img = blur_img.astype( 'uint8' )
    blur_img = cv2.equalizeHist( blur_img )
    _,thresh = cv2.threshold( blur_img , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    
    debugORSave(img, thresh, params,1,"1_BLURRING AND THRESHOLDING")
    
    return thresh

# Closing: Removing black spots from white regions. Basically Dilation followed by Erosion.
def Closing(img, params):
    
    input   = img
    kernel  = np.ones( ( params[ 'closing k size' ] , params[ 'closing k size' ] ) , np.uint8 )
    output  = cv2.morphologyEx( input , cv2.MORPH_CLOSE , kernel )
    
    debugORSave( input , output , params , 1 , "2_CLOSING" )
    
    return output

## Opening: Removing white spots from black regions. Basically Erosion followed by Dilation.
def Opening(img, params):

    input   = img
    kernel  = np.ones(( params[ 'opening k size' ] , params[ 'opening k size' ] ) , np.uint8 )
    output  = cv2.morphologyEx( input , cv2.MORPH_OPEN , kernel )
    
    debugORSave( input , output , params , 1 , "3_OPENING" )
    
    return output 

## Skeletonize Image
def Skeletonize(img, params): 

    input       = img 
    image       = invert( input / np.max( input ) )
    skeleton    = skeletonize( image )
    output      = ( skeleton / np.max( skeleton ) ) * 255 
    InvOutput   = invertBinaryImage( output )

    debugORSave( input , InvOutput , params , 1 , "4_SKELETONIZED" )
    
    return output

## Finds branching points for images with black background and white lines receive a degree matrix
def BreakBraches(img, params):

    input                               = np.copy( img )
    _, _, degrees                       = skeleton_to_csgraph( input )
    intersection_matrix                 = degrees > 2                   # consider all values larger than two as intersection
    input[intersection_matrix==True]    = 0
    InvInput                            = invertBinaryImage( input )
    
    debugORSave( img , InvInput , params , 0 , "5_BRANCHED SKELETON" )
    
    return input

## Skeleton Segmentation
def SkeletonSegmentation(img):
    
    input                       = np.copy( img )
    input                       = input.astype( 'uint8' )
    segmentedImg_temp           = label( input , connectivity = 2 )
    props                       = regionprops( segmentedImg_temp )

    return props

## Filtering out small backbones using the pixThresh variable and breaking them into uniform size.
def Filtered_Uniform_BB(img, obj_list, params):
    
    bb              = np.zeros( img.shape )
    residualFrac    = 0.3
    count           = 0
    minResidual     = int( residualFrac * params[ 'ellipse pixel size' ] )

    for i in range(len(obj_list)):
         
        BoneLen = len(obj_list[i].coords)
        if BoneLen > params['backbone threshold length']: #and BoneLen>minResidual:
            count += 1
            for ind,value in enumerate(obj_list[i].coords):
                
                if (ind+1)%params['ellipse pixel size'] == 0:
                    bb[value[0],value[1]] = 0
#                     if (BoneLen - (ind+1)) < minResidual:
#                         break
                else:
                    bb[value[0],value[1]] = 255
        
        BoneLen = 0

    Invbb = invertBinaryImage(bb)
    
    debugORSave( img , Invbb , params , 0 , "6_FILTERED AND UNIFORM SIZED BACKBONE" )
    
    return bb

# ## Filtering out small backbones using the pixThresh variable and breaking them into uniform size.
# def Remove_small_BB(img, obj_list, pixelThreshold):
#     bb = np.zeros(img.shape)
#     for i in range(len(obj_list)):
#         if len(obj_list[i]) > pixelThreshold:
#             for ind in obj_list[i]:
#                 bb[ind[0][1],ind[0][0]] = 255
#     return bb

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
def EllipseConstruction(img, params):

    input                       = np.copy(img)
    label_img                   = label(input)
    props                       = regionprops_table(label_img, properties = ( 'centroid' , 'orientation' , 'major_axis_length' , 'minor_axis_length' ) )
    props                       = pd.DataFrame( props )
    props['orientation']        = - 90 - props[ 'orientation' ] * 180/np.pi
    props['major_axis_length']  = props[ 'major_axis_length' ] / 2
    props['minor_axis_length']  = props[ 'minor_axis_length' ] / 2
    bb_props                    = pd.DataFrame()
    bb_props                    = props
    
    for ind in range(props.shape[0]):
        if props[ 'minor_axis_length' ][ ind ] < 1:
            bb_props = bb_props.drop([ind])   
        
        elif props[ 'major_axis_length' ][ ind ] / props[ 'minor_axis_length' ][ ind ] > params[ 'ellipse threshold aspect ratio' ]:
            ellip_temp_img = cv2.ellipse( input , ( int( props[ 'centroid-1' ][ ind ] ) , int( props[ 'centroid-0' ][ ind ] ) ) , \
                                         ( int( props[ 'major_axis_length' ][ ind ] ), int( props[ 'minor_axis_length' ][ ind ] ) ) , \
                                         int( props[ 'orientation' ][ ind ] ) , 0.0 , 360.0 , ( 255 , 0 , 0 ) , 2 );
        else: 
            bb_props = bb_props.drop([ind])
    
    bb_props_np         = bb_props.to_numpy()
    InvEllip_temp_img   = invertBinaryImage(ellip_temp_img)
    
    debugORSave( img , InvEllip_temp_img , params , 0 , "7_ELLIPSE INSCRIBED" )
    
    return ellip_temp_img, bb_props_np

## Creating Adjacency Matrix based on distance between centroid and the orientation angle
def AdjacencyMat(img, bb_props, params):# distanceThresh, thetaThresh):
    
    bb_props_np     = bb_props
    centroid_coord  = bb_props_np[ : , : 2 ]
    tree            = KDTree(centroid_coord, leaf_size=2)
    N               = len(bb_props_np)
    KNN_radius      = 2*params['ellipse pixel size'] + params['adjacency threshold distance']
    A_Mat           = np.zeros((N,N))

    for i in range(N):
        ind = tree.query_radius(np.reshape(centroid_coord[i],(1,2)), r=KNN_radius)  
        for j in ind[0]:
            if i == j:
                continue
            else:
                pts1        = majorAxisPoints( bb_props_np[ j ] )
                pts2        = majorAxisPoints( bb_props_np[ i ] )
                l2norm      = minDist( pts1 , pts2 )
                angleDiff   = abs( bb_props_np[ i ][ 2 ] - bb_props_np[ j ][ 2 ] )
                
                if l2norm < params['adjacency threshold distance'] and angleDiff < params['adjacency threshold angle']:
                    A_Mat[i][j] = 1
                    A_Mat[j][i] = 1
    
    A_Mat = np.maximum( A_Mat , A_Mat.transpose() )

    if params['show intermediate images'] == 1 or params['save intermediate images'] == 1:
        # Plotting the adjacent ellipse.
        A_Mat_img   = np.copy(img)
        for i in range(N):
            for j in range(N):
                if A_Mat[i][j] == 1:
                    poly        = bb_props_np[i]
                    A_Mat_img   = cv2.ellipse( A_Mat_img , ( int( poly[ 1 ] ) , int( poly[ 0 ] ) ) , ( int( poly[ 3 ] ) , int( poly[ 4 ] ) ) , poly[ 2 ] , 0.0 , 360.0 , ( 255 , 0 , 0 ) , 2 );
                    break
                    
        InvA_Mat_img = invertBinaryImage(A_Mat_img)
        debugORSave(img, InvA_Mat_img, params, 0, "8_ADJACENT ELLIPSES")
                
    return A_Mat

## Returns polymer Major axis endpoint and the midpoints
def majorAxisPoints(poly):
    
    temp            = np.zeros([3,2])
    ang             = poly[2]*np.pi/180
    
    temp[ 0 , 0 ]   = int( poly[1] - poly[3] * np.cos( ang ) )  ## Major axis end point 1
    temp[ 0 , 1 ]   = int( poly[0] - poly[3] * np.sin( ang ) )
    
    temp[ 1 , 0 ]   = poly[1]                                   ## centroid
    temp[ 1 , 1 ]   = poly[0]
    
    temp[ 2 , 0 ]   = int(poly[1] + poly[3]*np.cos(ang))        ## Major axis end point 2
    temp[ 2 , 1 ]   = int(poly[0] + poly[3]*np.sin(ang))
    
    return temp

## Return minimum distance b/w two sets of points
def minDist(pts1, pts2):

    minD = np.linalg.norm(pts1[0]-pts2[0]) 
    
    for i in range(len(pts1)):
        for j in range(len(pts2)):
            temp = np.linalg.norm(pts1[i]-pts2[j])
            if temp < minD:
                minD = temp
    return minD

## Connected Components:
def ConnecComp(img, A_Mat, props, params):

    polyProps   = props 
    N           = A_Mat.shape[0]
    visited     = np.zeros(N)
    cc          = []
    
    for v in range(N):
        if visited[v] == 0:
            temp = []
            cc.append(DFSUtil(temp, v, visited, N, A_Mat))
    
    ccImg                   = np.copy(img)
    AllClusterPointCloud    = []
    crystalAngles           = []

    """ startAngle = 0
    endAngle = 360
    color = (255, 0, 0)
    thickness = 2 """

    for i in range(len(cc)):
        if len(cc[i]) >= params['cluster threshold size']:
            majorsAxisPointCloud    = []
            ellipseAngels           = []

            for j in cc[i]:
                poly                        = polyProps[j]
                ccImg                       = cv2.ellipse(ccImg, (int(poly[1]), int(poly[0])), (int(poly[3]), int(poly[4])), poly[2], 0.0, 360.0, (255, 0, 0), 2);
                temp                        = majorAxisPoints(poly)
                temp[temp<0]                = 0
                temp[temp>=img.shape[0]]    = img.shape[0]-1
                temp                        = temp.astype('int32')
                #ccImg                      = cv2.circle(ccImg, (int(temp[0,0]),int(temp[0,1])), radius=5, color=(255, 0, 0), thickness=-1)
                #ccImg                      = cv2.circle(ccImg, (int(temp[1,0]),int(temp[1,1])), radius=5, color=(255, 0, 0), thickness=-1)
                
                majorsAxisPointCloud.append(temp[0,:])
                majorsAxisPointCloud.append(temp[2,:])
                ellipseAngels.append(poly[2])
            AllClusterPointCloud.append(majorsAxisPointCloud)
            crystalAngles.append(mean(ellipseAngels))
    
    InvCcImg        = invertBinaryImage(ccImg)
    debugORSave(img, InvCcImg, params, 0, "9_CLUSTERS")
    
    return ccImg, AllClusterPointCloud, crystalAngles

## Connected Components: 
def DFSUtil(temp, v, visited, numEllipse, adjacencyMat):
    N           = numEllipse
    visited[v]  = 1             # Mark the current vertex as visited
    temp.append(v)              # Store the vertex to list 
    
    # Repeat for all vertices adjacent
    for i in range(N):
        if adjacencyMat[v][i] == 1:
            if visited[i] == 0:           
                temp = DFSUtil(temp, i, visited, N, adjacencyMat)   # Update the list
    #print(temp)
    return temp

def PlottingAndSaving(img, ClusterPointCloud, ProjectPath, ResultDir, ImgName, crystalAng, params):
    RGBImg          = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # CrystalImg      = img       #invertBinaryImage(finalImg)
    
    figure, axes    = plt.subplots(nrows=1, ncols=2, figsize = (50,25))
    axes[1].imshow(RGBImg, vmin=0, vmax=255)
    
    unitConv_pixSq2nmSq = 1/params['pix to nm']**2
    crystalArea         = []
    centroid            = []
    boundingBox         = []
    color               = ['b', 'g', 'r', 'c', 'm','y','w']

    for ind, cluster in enumerate(ClusterPointCloud):
        PointCloud  = np.array(cluster)
        hull        = ConvexHull(PointCloud)
        
        #Get centoid
        cx          = np.mean(hull.points[hull.vertices,0], dtype= int)
        cy          = np.mean(hull.points[hull.vertices,1], dtype= int)
        centroid.append([cx, cy])
        
        # Get Boundingbox coordinates
        x_minMax    = [np.amin(hull.points[hull.vertices,0]), np.amax(hull.points[hull.vertices,0])]
        y_minMax    = [np.amin(hull.points[hull.vertices,1]), np.amax(hull.points[hull.vertices,1])]
        boundingBox.append( [ int( x_minMax[0] ) , int( y_minMax[0] ) , int( x_minMax[1] ) , int( y_minMax[1] ) ] )
        
        c       = random.choice(color)
        arrLen  = min( int( x_minMax[1] - x_minMax[0] ) / 2, int( y_minMax[1] - y_minMax[0] ) / 2 ) / 3
        plt.arrow( cx , cy , arrLen * np.cos( crystalAng[ ind ] * np.pi/180 ) , arrLen * np.sin( crystalAng[ind] * np.pi/180 ) , linewidth = 7.0 , color = c )

        ################### Code for plotting Convex Hull #####################################
        for simplex in hull.simplices:
            plt.plot( PointCloud[ simplex , 0 ] , PointCloud[ simplex , 1 ] , linewidth = 7.0 , color = c )
        ########################################################################################
        
        ##################### Code for Plotting Alpha Shape (Tight fitting boundry)#############
        alpha_shape     = alphashape.alphashape( PointCloud , alpha = 0.005 )
        crystalArea.append( ( alpha_shape.area ) * unitConv_pixSq2nmSq )
        axes[1].add_patch( PolygonPatch( alpha_shape , alpha = 0.2 ) )
        ########################################################################################

    axes[0].imshow( img , cmap = 'gray')
    figure.tight_layout()
    figure.savefig( join(ProjectPath , ResultDir , ImgName[:-4]+'.png') )

    if params['show final image'] == 1:
        plt.show()
    figure.clf()
    plt.close()
    plt.clf()
    gc.collect()
    df_BB = pd.DataFrame( list( zip( boundingBox ) ) , columns = [ "Top Left(x_y) Bottom Right(x_y)" ] )
    
    return crystalArea, centroid, df_BB

def evaluateDspacing(bb, img, params):
    
    bb_arr              = bb.to_numpy()
    peak_cutoff_factor  = 1.15 
    if bb_arr.size == 0:
        return []
    
    bb_arr      = np.hstack(bb_arr)
    d_spaces    = []

    for region in range(len(bb_arr)):
        
        img_cropped     = img[ bb_arr[ region ][ 1 ] : bb_arr[ region ][ 3 ] , bb_arr[ region ][ 0 ] : bb_arr[ region ][ 2 ] ]
        smallerDim      = min(img_cropped.shape)

        row_low         = int( ( img_cropped.shape[0] - smallerDim ) / 2 )
        row_high        = int( ( img_cropped.shape[0] + smallerDim ) / 2 )
        col_low         = int( ( img_cropped.shape[1] - smallerDim ) / 2 )
        col_high        = int( ( img_cropped.shape[1] + smallerDim ) / 2 )

        square_region   = img_cropped[ row_low : row_high, col_low : col_high ]
        img_centre      = (np.asarray(square_region.shape)/2).astype(int)
        
        f               = np.fft.fft2( square_region )
        fshift          = np.fft.fftshift( f )
        power_spectrum  = 1 * np.log( np.abs( fshift ) + 1 )
        
        # plot3d(power_spectrum)

        arrSiz          = np.asarray( power_spectrum.shape )
        qIn, qOut       = ringSize( arrSiz , params )
        TFring, nonZero = draw_ring( shape = power_spectrum.shape , rIn = qIn , rOut = qOut )
        power_spectrum  = filter_ring( TFring , power_spectrum )

        bandpass_pow_spec_mean      = np.sum( power_spectrum ) / nonZero
        
        # plt.imshow(power_spectrum, cmap='hot', interpolation='nearest')
        # plt.show()
        
        """ ind             = np.unravel_index( np.argmax( power_spectrum , axis = None ) , power_spectrum.shape )

        ps2             = power_spectrum
        ps2[ind]        = 0
        ind2            = np.unravel_index( np.argmax( ps2 , axis = None ) , ps2.shape )

        # plt.imshow(ps2, cmap='hot', interpolation='nearest')
        # plt.show()

        # freq_coord      = ind - ( arrSiz / 2 )
        freq_coord      = (np.asarray(ind) - np.asarray(ind2))
        freq_coord      = freq_coord.astype( np.int32 )
        
        if np.all( ( freq_coord == 0 ) ):
            freq_coord  = np.ones( ( 2 ) )
        
        freq            = np.linalg.norm( freq_coord )/2
        tp              = 1 / freq
        # orientedLen     = np.abs( imgLength( freq_coord , arrSiz ) )
        # d_space         = tp * orientedLen
        d_space         = tp * arrSiz[0] """

        
        ### Setting a cuttoff for the max of power spectrum to be considered as peak.  
        if np.max( power_spectrum ) >= peak_cutoff_factor * bandpass_pow_spec_mean:
            ind             = np.unravel_index( np.argmax( power_spectrum , axis = None ) , power_spectrum.shape )
            
            ps2             = power_spectrum
            ps2[ind]        = 0
            ind2            = np.unravel_index( np.argmax( ps2 , axis = None ) , ps2.shape )

            freq_coord      = (np.asarray(ind) - np.asarray(ind2))
            freq_coord      = freq_coord.astype( np.int32 )
        
            if np.all( ( freq_coord == 0 ) ):
                freq_coord = np.ones( ( 2 ) )
            
            freq            = np.linalg.norm( freq_coord )/2
            tp              = 1 / freq
            # orientedLen     = np.abs( imgLength( freq_coord , arrSiz ) )
            d_space         = tp * arrSiz[0]
        else:
            d_space = 0

        d_spaces.append( d_space / params[ 'pix to nm' ] )

        print("\n")
        # print("size of Arr      :", arrSiz)
        print("Max Magnitude    :", np.max(power_spectrum))
        print("Band Pass Mean   :", bandpass_pow_spec_mean)
        # print("1st Freq Index   :", ind)
        # print("2nd Freq Index   :", ind2)
        # print("freq vector      :", freq_coord)
        # print("freq             :", freq)
        # print("Time Period      :", tp)
        # # print("Oriented Length  :", orientedLen)
        # print("D space in px    :", d_space)
        print("D space in nm    :", d_space/params['pix to nm'])

        # plot3d(power_spectrum)

        """ plt.subplot(121)
        plt.imshow(img_cropped, cmap = 'gray')
        plt.title('Input Image')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(power_spectrum, cmap = 'gray')
        plt.title('Power Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
        plt.close()
        plt.clf() """
    return d_spaces

def draw_ring(shape,rIn, rOut):
    rows, cols = shape[0], shape[1]
    crow, ccol = (shape[0]) / 2, (shape[1]) / 2

    f_low_pixels    = rIn  #get_q_pixels(q_low, M, dx)
    f_high_pixels   = rOut #get_q_pixels(q_high, M, dx)

    mask            = np.zeros((rows, cols), dtype=np.bool)
    center          = [crow, ccol]
    x, y            = np.ogrid[:rows, :cols]
    
    mask_area       = np.logical_and(   ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= f_low_pixels ** 2),
                                        ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= f_high_pixels ** 2))
    mask[mask_area] = True
    count           = np.sum(mask)

    return mask, count

def filter_ring(TFring,fft_img_channel):
    
    temp            = np.zeros( fft_img_channel.shape[ : 2 ] )
    temp[ TFring ]  = fft_img_channel[ TFring ]
    
    return(temp)

def ringSize(arrSiz, params):
    f_in_nm, f_out_nm = 0, 0
    f_in_px, f_out_px = 0, 0 
    
    if 1.8 <= params[ 'd space nm' ] <= 2.2 :
        lowDS       = 1.8       ## nm 
        higherDS    = 2.5       ## nm
    
    elif 0.5 <= params[ 'd space nm' ] <= 1.5 :
        lowDS       = 0.5       ## nm 
        higherDS    = 1.5       ## nm
    
    elif 0.1 <= params[ 'd space nm' ] <= 0.5 : 
        lowDS       = 0.1       ## nm 
        higherDS    = 0.5       ## nm

    f_in_nm     = 1 / higherDS
    f_out_nm    = 1 / lowDS

    f_in_px     = arrSiz[0] * f_in_nm / params[ 'pix to nm' ]
    f_out_px    = np.linalg.norm( arrSiz ) * f_out_nm / params[ 'pix to nm' ]

    return f_in_px, f_out_px

def plot3d(arr):
    nx,ny = arr.shape[1], arr.shape[0]
    x = range(nx)
    y = range(ny)
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D

    ha.plot_surface(X, Y, arr, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()