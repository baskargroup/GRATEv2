Config file example:

## Directory paths
data_dir         = "DATA/sampleData/";     # Data directory wrt project path 
base_result_dir  = "Results/temp/";        # Base result directory wrt project path

## Parameters
dspace_nm   = [1.9];                # The D-Spacing value (eg. 1.9, 0.7) at which the algorithm will run.
pix_2_nm    = 78.5;                 # Image resolution, number of pixels per nanometer

blur_iteration          = 15;       # Number of Blur Iteration
Blur_kernel_propCons    = 0.15;     # Proportionality constant of d-spacing (in pixel) for the blur kernel size
closing_k_size          = 15;       # Closing Kernel Size
opening_k_size          = 17;       # Opening Kernel Size
pixThresh_propCons      = 0.625;    # Proportionality constant of d-spacing (in pixel) for the threshold number of pixels consituting Backbone
ellipse_len_propCons    = 1.5;      # Proportionality constant of d-spacing (in pixel) for the breaking Backbone into uniform size before constructing ellipse
ellipse_aspect_ratio    = 5;        # Threshold ellipse aspect Ratio 
thresh_dist_propCons    = 2;        # Proportionality constant of d-spacing (in pixel) for the distance threshold for adjacency matrix
thresh_theta            = 10;       # delta Theta threshold for adjacency matrix
cluster_size            = 7;        # Threshold ellipse in Crystal cluster
dspace_bandpass          = 0.2;     # Bandpass filter range across d-spacing
powSpec_peak_thresh     = 1.15;     # 1.20 works for all
thresh_area_factor      = 4;        # Cut off area factor of d-spacing^2

## Modes
debug               = 1;            # To run on single image and save intermediate steps
save_BB             = 1;            # To save Bounding box coordinates: 1, Not to: 0
save_backbone_coords= 1;            # To save backbone coordinates: 1, Not to: 0
result_display      = 0;            # To display final result in notebook: 1, Not to: 0
image_scale_percent = 50;           # Scaling the image before display

------------------------------------

# Segregating config paras to find relevant ones for the BO algorithm, fixing the others

## Irrelevant paras (Kept constant)

### Directory paths
data_dir         = "DATA/sampleData/";     # Data directory wrt project path
base_result_dir  = "Results/temp/";        # Base result directory wrt project path

### Modes
debug               = 1;            # To run on single image and save intermediate steps
save_BB             = 1;            # To save Bounding box coordinates: 1, Not to: 0
save_backbone_coords= 1;            # To save backbone coordinates: 1, Not to: 0
result_display      = 0;            # To display final result in notebook: 1, Not to: 0
image_scale_percent = 50;           # Scaling the image before display

### Parameters

pix_2_nm    = 78.5;                 # Image resolution, number of pixels per nanometer

## Relevant paras (Optimized by BO)

### Parameters

dspace_nm               = [1.9];    # (Real) The D-Spacing value (eg. 1.9, 0.7) at which the algorithm will run.
blur_iteration          = 15;       # (Integer) Number of Blur Iteration
Blur_kernel_propCons    = 0.15;     # (Real) Proportionality constant of d-spacing (in pixel) for the blur kernel size
closing_k_size          = 15;       # (Integer) Closing Kernel Size
opening_k_size          = 17;       # (Integer) Opening Kernel Size
pixThresh_propCons      = 0.625;    # (Real) Proportionality constant of d-spacing (in pixel) for the threshold number of pixels consituting Backbone
ellipse_len_propCons    = 1.5;      # (Real) Proportionality constant of d-spacing (in pixel) for the breaking Backbone into uniform size before constructing ellipse
ellipse_aspect_ratio    = 5;        # (Real) Threshold ellipse aspect Ratio
thresh_dist_propCons    = 2;        # (Real) Proportionality constant of d-spacing (in pixel) for the distance threshold for adjacency matrix
thresh_theta            = 10;       # (Real) delta Theta threshold for adjacency matrix
cluster_size            = 7;        # (Integer) Threshold ellipse in Crystal cluster
dspace_bandpass         = 0.2;      # (Real) Bandpass filter range across d-spacing
powSpec_peak_thresh     = 1.15;     # (Real) 1.20 works for all
thresh_area_factor      = 4;        # (Real) Cut off area factor of d-spacing^2