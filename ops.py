
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import ConvexHull
from utils import   getCentroid, \
                    getBoundingBox, \
                    getAlphaShape, \
                    getCrystalSizeAndOrientation, \
                    getAngleDifference, \
                    pltAlphaShape, \
                    pltOrientationLine, \
                    pltConvexHull, \
                    isAreaSmall
from os.path import join
from matplotlib import cm
from skimage import exposure

plt.ioff()

############### Operations ############################3

def histEq(img, runtype = 1):
    # Hist Eq Type 1 
    if runtype == 1:
        img = img.astype('uint8')
        img = cv2.equalizeHist(img)
    
    # Hist Eq Type 2
    elif runtype == 2:
        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 2 - 1
        img_eq = exposure.equalize_adapthist(img_normalized, clip_limit=0.03)
        img = (img_eq + 1) / 2
        
    else:
        print('Invalid Hist Eq Type')
    return img

################ SCIPY BASED ELLIPSE CONSTRUCTION FUNCTION ###############################
# @njit
def majorAxisPoints(poly):
    
    temp            = np.zeros([3,2])
    ang             = poly[2]*np.pi/180
    cos_ang, sin_ang = np.cos(ang), np.sin(ang)
    delta_x = poly[3] * cos_ang
    delta_y = poly[3] * sin_ang
    
    temp[ 0 , 0 ]   = int( poly[1] - delta_x )  ## Major axis end point 1
    temp[ 0 , 1 ]   = int( poly[0] - delta_y )
    
    temp[ 1 , 0 ]   = poly[1]                                   ## centroid
    temp[ 1 , 1 ]   = poly[0]
    
    temp[ 2 , 0 ]   = int(poly[1] + delta_x)        ## Major axis end point 2
    temp[ 2 , 1 ]   = int(poly[0] + delta_y)
    
    return temp

def initialize_plot(last_dspace_run, 
                    BayesianOptRun,
                    RGBImg_shape = None,
                    downsample_factor = 1):
    
    if BayesianOptRun == True:
        numSubplots = 1
        assert RGBImg_shape is not None, "RGBImg_shape must be provided for single subplot"
        orig_img_plt_idx = 0
        result_img_plt_idx = 0
        fig, axes = plt.subplots(nrows=1, 
                                 ncols=1, 
                                 figsize=(RGBImg_shape[1] / (100 * downsample_factor), 
                                          RGBImg_shape[0] / (100 * downsample_factor)))
        axes.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0)
        return (fig, axes, orig_img_plt_idx, result_img_plt_idx)
    
    else:
        if last_dspace_run:
            numSubplots = 2
            orig_img_plt_idx = 0
            result_img_plt_idx = 1
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(50, 25))
            # axes.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.tight_layout(pad=0)
            return (fig, axes, orig_img_plt_idx, result_img_plt_idx)
        else:
            numSubplots = 1
            assert RGBImg_shape is not None, "RGBImg_shape must be provided for single subplot"
            orig_img_plt_idx = 0
            result_img_plt_idx = 0
            fig, axes = plt.subplots(nrows=1, 
                                     ncols=1, 
                                     figsize=(RGBImg_shape[1] / (100 * downsample_factor), 
                                              RGBImg_shape[0] / (100 * downsample_factor)))
            axes.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.tight_layout(pad=0)
            return (fig, axes, orig_img_plt_idx, result_img_plt_idx)
        

def process_cluster(OrigImg, 
                    cluster, 
                    crystal_ang, 
                    params, 
                    color):
    point_cloud = np.array(cluster)
    hull = ConvexHull(point_cloud)
    
    cx, cy      = getCentroid(hull)
    (x_min_max, 
     y_min_max) = getBoundingBox(hull)
    d_space     = evaluateDspacing( OrigImg, 
                                    params, 
                                    x_min_max, 
                                    y_min_max)
    alpha_shape = getAlphaShape(point_cloud)

    if d_space == 0 or isAreaSmall(alpha_shape.area, params):
        return None

    major_len, minor_len, major_axes_angle = getCrystalSizeAndOrientation(hull)
    angle_diff_val = getAngleDifference(major_axes_angle, 
                                        crystal_ang)

    crystal_properties = {
        "centroid": [cx, cy],
        "alpha_shape": alpha_shape,
        "x_min_max": x_min_max,
        "y_min_max": y_min_max,
        "area": alpha_shape.area * (1 / params['pix to nm'] ** 2),
        "angle": crystal_ang,
        "d_space": d_space,
        "major_len": major_len * (1 / params['pix to nm']),
        "minor_len": minor_len * (1 / params['pix to nm']),
        "major_axes_angle": major_axes_angle,
        "angle_diff": angle_diff_val,
        "color": color,
        "hull": hull,
        "point_cloud": point_cloud
    }
    
    if params['save bounding box'] == 1:
        crystal_properties['bounding_box'] = [int(x_min_max[0]), 
                                              int(y_min_max[0]), 
                                              int(x_min_max[1]), 
                                              int(y_min_max[1])]
    
    return crystal_properties
    
def plot_results(last_dspace_run,
                 axes, 
                 results, 
                 orig_img, 
                 RGB_img, 
                 orig_img_plt_idx, 
                 result_img_plt_idx, 
                 BayesianOptRun):
    if BayesianOptRun == True:
        axes.imshow(RGB_img, vmin=0, vmax=255)
            
        for result in results:
            if result:
                pltConvexHull(axes, 
                              result['hull'], 
                              result['point_cloud'], 
                              result['color'])
    
    else:
        if not last_dspace_run:
            axes.imshow(RGB_img, vmin=0, vmax=255)
            for result in results:
                if result:
                    pltAlphaShape(axes, 
                                  result['alpha_shape'])
                    pltOrientationLine(axes, 
                                       result['angle'], 
                                       result['color'], 
                                       result['x_min_max'], 
                                       result['y_min_max'], 
                                       result['centroid'][0], 
                                       result['centroid'][1])
                    pltConvexHull(axes, 
                                  result['hull'], 
                                  result['point_cloud'], 
                                  result['color'])
        
        else:
            axes[result_img_plt_idx].imshow(RGB_img, vmin=0, vmax=255)
            
            for result in results:
                if result:
                    pltAlphaShape(axes[result_img_plt_idx], 
                                  result['alpha_shape'])
                    pltOrientationLine(axes[result_img_plt_idx], 
                                       result['angle'], 
                                       result['color'], 
                                       result['x_min_max'], 
                                       result['y_min_max'], 
                                       result['centroid'][0], 
                                       result['centroid'][1])
                    pltConvexHull(axes[result_img_plt_idx], 
                                  result['hull'], 
                                  result['point_cloud'], 
                                  result['color'])
                    # Other plotting based on 'result'

            axes[orig_img_plt_idx].imshow(orig_img, cmap='gray')

def extract_results(processed_clusters):
    """
    Extracts and aggregates results from processed clusters.

    Args:
        processed_clusters (list): List of dictionaries containing processed cluster data.

    Returns:
        tuple: Aggregated results including areas, centroids, angles, etc.
    """
    crystal_area = []
    centroid = []
    crystal_angles_final = []
    d_spaces = []
    bounding_box = []
    crystal_major_axis_length = []
    crystal_minor_axis_length = []
    crystal_major_axis_angle = []
    angle_difference = []

    for result in processed_clusters:
        if result is not None:
            crystal_area.append(result['area'])
            centroid.append(result['centroid'])
            crystal_angles_final.append(result['angle'])
            d_spaces.append(result['d_space'])
            crystal_major_axis_length.append(result['major_len'])
            crystal_minor_axis_length.append(result['minor_len'])
            crystal_major_axis_angle.append(result['major_axes_angle'])
            angle_difference.append(result['angle_diff'])

            if 'bounding_box' in result:
                bounding_box.append(result['bounding_box'])

    return (crystal_area, centroid, crystal_angles_final, d_spaces, bounding_box,
            crystal_major_axis_length, crystal_minor_axis_length,
            crystal_major_axis_angle, angle_difference)

def evaluateDspacing(Entire_img, params, x_minMax, y_minMax):
    
    img = Entire_img[int( y_minMax[0] ) : int( y_minMax[1] ), int( x_minMax[0] ) : int( x_minMax[1] )]
    peak_cutoff_factor  = params[ 'pow spec peak vs mean factor' ]
    smallerDim      = min(img.shape)

    row_low         = int( ( img.shape[0] - smallerDim ) / 2 )
    row_high        = int( ( img.shape[0] + smallerDim ) / 2 )
    col_low         = int( ( img.shape[1] - smallerDim ) / 2 )
    col_high        = int( ( img.shape[1] + smallerDim ) / 2 )

    square_region   = img[ row_low : row_high, col_low : col_high ]
    f               = np.fft.fft2( square_region )
    fshift          = np.fft.fftshift( f )    
    power_spectrum  = 1 * np.log( np.abs( fshift ) + 1 )
    
    # plot3d(power_spectrum)

    arrSiz          = np.asarray( power_spectrum.shape )
    qIn, qOut       = ringSize( arrSiz , params )
    TFring, nonZero = draw_ring( shape = power_spectrum.shape , rIn = qIn , rOut = qOut )
    power_spectrum  = filter_ring( TFring , power_spectrum )
    ps_maxMag       = np.max(power_spectrum)
    bandpass_pow_spec_mean      = np.sum( power_spectrum ) / nonZero
    if ps_maxMag >= peak_cutoff_factor * bandpass_pow_spec_mean:
        
        ind             = np.unravel_index( np.argmax( power_spectrum , axis = None ) , power_spectrum.shape )
        freq_coord      = (np.asarray(ind) - arrSiz/2)
        freq_coord      = freq_coord.astype( np.int32 )
    
        if np.all( ( freq_coord == 0 ) ):
            freq_coord = np.ones( ( 2 ) )
        
        freq            = np.linalg.norm( freq_coord )
        # freq            = np.linalg.norm( freq_coord )/2
        tp              = 1 / freq
        d_space         = tp * arrSiz[0]
        d_space         = d_space / params[ 'pix to nm' ]
    
    else:
        d_space = 0    

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

    return d_space


def draw_ring(shape,rIn, rOut):
    rows, cols = shape[0], shape[1]
    crow, ccol = (shape[0]) / 2, (shape[1]) / 2

    f_low_pixels    = rIn
    f_high_pixels   = rOut

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
    
    # Check if params['d space band'] exists in params
    if 'd space bandpass' in params:
        lowDS       = (1 - params['d space bandpass']) * params['d space nm']       ## nm
        higherDS    = (1 + params['d space bandpass']) * params['d space nm']       ## nm
    
    else: # Default case 20% range
        lowDS       = 0.8 * params[ 'd space nm' ]       ## nm 
        higherDS    = 1.2 * params[ 'd space nm' ]       ## nm

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