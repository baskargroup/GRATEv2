from utils import invertBinaryImage, timeit, minDist, load_img_result_dir, create_rgb_image
from pathlib import Path
from ops import histEq, majorAxisPoints, process_cluster, initialize_plot, plot_results, extract_results
import cv2
import time
from matplotlib import pyplot as plt
from os.path import join
from skimage import io
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from sklearn.neighbors import KDTree
from skan import skeleton_to_csgraph
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd 

import random
import gc

import pickle
from skimage.filters import threshold_otsu
import numpy as np

plt.ioff()

class ImageProcessor:
    def __init__(self, img_path, parameters, last_run):
        self.parameters = parameters
        self.img_path = img_path
        self.img = io.imread(img_path).astype('float64')
        self.last_run = last_run
        
    @timeit
    def GRATE(self):
        
        self.save_image_to_result_dir()
            
        thresh = self.BlurThresh()
        
        skeleton = self.process_skeleton(thresh)
        
        skeleton = self.BreakBranches(skeleton)
        
        temp = self.SkeletonSegmentation(skeleton)
        
        Broken_backbone_img = self.Filtered_Uniform_BB( temp)
        
        bb_ellipse_props = self.EllipseConstruction(Broken_backbone_img)
        
        adjacencyMat = self.AdjacencyMat(Broken_backbone_img, bb_ellipse_props)
        
        AllClusterPointCloud, crystalAngles = self.ConnecComp(Broken_backbone_img, adjacencyMat, bb_ellipse_props)
        
        crystalArea, centroid, crystalAngles_final, dspaces, df_boundBox, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference = self.PlottingAndSaving( AllClusterPointCloud, crystalAngles, self.last_run)
        
        df = self.process_and_save_dataframe(centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference)
        
        self.process_save_backbone_coords(df_boundBox)
        return df
    
    @timeit
    def save_image_to_result_dir(self):
        """Save image to result directory."""
        
        if (self.parameters['result image directory'] / (self.img_path.stem + self.parameters['save image format'])).is_file():
            print("Image already present in the result image directory")
        else:
            print("Saving image to result image directory")
            cv2.imwrite(str(self.parameters['result image directory'] / (self.img_path.stem + self.parameters['save image format'])), \
                cv2.cvtColor(self.img.astype('uint8'), cv2.COLOR_GRAY2RGB))
        
    @timeit
    def BlurThresh(self):
        blur_img    = self.img
        k_size      = self.parameters['blur k size']
        
        if k_size%2 != 1:
            k_size = k_size+1

        for i in range( self.parameters[ 'blur iterations' ] ):
            blur_img = cv2.blur( blur_img , ( k_size , k_size ) )

        blur_img = histEq(blur_img)
        thresh = threshold_otsu(blur_img)
        binary = blur_img > thresh
        
        self.debugORSave(self.img, binary, 1, "1_BLURRING AND THRESHOLDING")
        return binary

    def process_skeleton(self, thresh):
        if self.parameters['d space pix'] < 78:
            result = self.Skeletonize(thresh)
        else:
            closing = self.Closing(thresh)
            opening = self.Opening(closing)
            result = self.Skeletonize(opening) 
        return result
    
    
    @timeit
    def BreakBranches(self, img):
        '''
        Finds branching points for images with black background and white lines receive a degree matrix
        '''
        input                               = np.copy( img )
        _, _, degrees                       = skeleton_to_csgraph( input )
        intersection_matrix                 = degrees > 2                   # consider all values larger than two as intersection
        input[intersection_matrix==True]    = 0
        InvInput                            = invertBinaryImage( input )
        
        self.debugORSave( img , InvInput , 0,"5_BRANCHED SKELETON" )
        return input
    
    @timeit
    def SkeletonSegmentation(self, img):
        '''
        Skeleton Segmentation
        '''
        input                       = np.copy( img )
        input                       = input.astype( 'uint16' )
        segmentedImg_temp           = label( input , connectivity = 2 )
        props                       = regionprops( segmentedImg_temp )
        return props
    
    @timeit
    def Filtered_Uniform_BB(self, obj_list):
        '''
        Filtering out small backbones using the pixThresh variable and breaking them into uniform size.
        '''
        bb              = np.zeros( self.img.shape )
        # residualFrac    = 0.3
        # count           = 0
        # minResidual     = int( residualFrac * params[ 'ellipse pixel size' ] )

        for i in range(len(obj_list)):
            
            BoneLen = len(obj_list[i].coords)
            if BoneLen > self.parameters['backbone threshold length']: #and BoneLen>minResidual:
                # count += 1
                for ind,value in enumerate(obj_list[i].coords):
                    
                    if (ind+1)%self.parameters['ellipse pixel size'] == 0:
                        bb[value[0],value[1]] = 0
    #                     if (BoneLen - (ind+1)) < minResidual:
    #                         break
                    else:
                        bb[value[0],value[1]] = 255
            
            BoneLen = 0

        Invbb = invertBinaryImage(bb)        
        self.debugORSave( self.img , Invbb, 0, "6_FILTERED AND UNIFORM SIZED BACKBONE" )
        return bb
    
    @timeit
    def EllipseConstruction(self, input_img):

        label_img   = label(input_img)  
        props       = regionprops_table(label_img, properties = ( 'centroid' , \
            'orientation' , 'major_axis_length' , 'minor_axis_length', 'coords' ) )
        props       = pd.DataFrame( props )
        
        props['orientation']        = - 90 - props[ 'orientation' ] * 180/np.pi
        props['major_axis_length'] /= 2
        props['minor_axis_length'] /= 2
        
        # Filter ellipses based on aspect ratio
        aspect_ratio_threshold = self.parameters['ellipse threshold aspect ratio']
        filtered_props = props[(props['minor_axis_length'] >= 1) & 
                               ((props['major_axis_length'] / props['minor_axis_length']) >= aspect_ratio_threshold)]
        
        
        
        if self.parameters['debug'] == 1:
            # Draw ellipses on the image
            ellip_temp_img = np.copy(input_img)
            for _, row in filtered_props.iterrows():
                ellip_temp_img = cv2.ellipse(
                    ellip_temp_img, 
                    (int(row['centroid-1']), int(row['centroid-0'])),
                    (int(row['major_axis_length']), int(row['minor_axis_length'])),
                    int(row['orientation']), 0.0, 360.0, (255, 0, 0), 2
                )
            InvEllip_temp_img   = invertBinaryImage(ellip_temp_img)
            
            self.debugORSave( input_img , InvEllip_temp_img , 0, "7_ELLIPSE INSCRIBED" )
        return filtered_props.to_numpy()
    
    @timeit
    def AdjacencyMat(self, img, bb_props_np):
        '''
        Creating Adjacency Matrix based on distance between centroid and the orientation angle
        '''
        centroid_coord  = bb_props_np[ : , : 2 ]
        N               = len(bb_props_np)
        
        if N == 0:
            print("No ellipses found")
            return np.zeros((N,N))
        
        tree            = KDTree(centroid_coord, leaf_size=2)
        KNN_radius      = 2*self.parameters['ellipse pixel size'] + self.parameters['adjacency threshold distance']
        A_Mat           = np.zeros((N,N))
        
        # Pre-calculate major axis points for all props
        all_major_axis_points = np.array([np.array(majorAxisPoints(prop[:4])).astype('float32') for prop in bb_props_np])

        for i in range(N):
            ind = tree.query_radius(np.reshape(centroid_coord[i],(1,2)), r=KNN_radius)  
            for j in ind[0]:
                if i >= j: # To avoid double counting
                    continue
                l2norm      = minDist( all_major_axis_points[j] , all_major_axis_points[i] )
                angleDiff   = abs( bb_props_np[ i ][ 2 ] - bb_props_np[ j ][ 2 ] )
                
                if l2norm < self.parameters['adjacency threshold distance'] and angleDiff < self.parameters['adjacency threshold angle']:
                    A_Mat[i][j] = A_Mat[j][i] = 1
        
        A_Mat = np.maximum( A_Mat , A_Mat.transpose() )

        if self.parameters['debug'] == 1:
            # Plotting the adjacent ellipse.
            A_Mat_img   = np.copy(img)
            for i in range(N):
                for j in range(N):
                    if A_Mat[i][j] == 1:
                        poly        = bb_props_np[i]
                        A_Mat_img   = cv2.ellipse( A_Mat_img , ( int( poly[ 1 ] ) , int( poly[ 0 ] ) ) , ( int( poly[ 3 ] ) , int( poly[ 4 ] ) ) , poly[ 2 ] , 0.0 , 360.0 , ( 255 , 0 , 0 ) , 2 );
                        break
                        
            InvA_Mat_img = invertBinaryImage(A_Mat_img)
            self.debugORSave(img, InvA_Mat_img, 0, "8_ADJACENT ELLIPSES")        
        return A_Mat
    
    def group_nodes_by_component_efficient(self, labels):
        """Group nodes by component using numpy."""
        unique_components = np.unique(labels)
        return [np.where(labels == component)[0] for component in unique_components]
    
    def process_component(self, component, polyProps, ccImg):
        '''
        Process individual connected components.
        '''
        majorsAxisPointCloud = []
        ellipseAngles = []

        for index in component:
            poly = polyProps[index]
            temp = majorAxisPoints(poly)
            temp = np.clip(temp, 0, ccImg.shape[0] - 1).astype('int32')
            majorsAxisPointCloud.extend([temp[0, :], temp[2, :]])
            ellipseAngles.append(poly[2])
        
        for index in component:
            poly = polyProps[index]
            ccImg = cv2.ellipse(ccImg, (int(poly[1]), int(poly[0])), (int(poly[3]), int(poly[4])), poly[2], 0.0, 360.0, (255, 0, 0), 2);

        return majorsAxisPointCloud, ellipseAngles

    def save_backbone_coords(self, backboneCoords):
        '''
        Save backbone coordinates if the parameter is set.
        '''
        if self.parameters['save backbone coords'] == 1:
            filepath = join(self.parameters['result backbone coords'], self.img_path.stem + '.pickle')
            with open(filepath, "wb") as filehandler:
                pickle.dump(backboneCoords, filehandler)
    
    @timeit
    def ConnecComp(self, img, A_Mat, polyProps):
        '''
        Analyze connected components based on adjacency matrix and properties of ellipses.
        '''
        sparseA_Mat = sp.csr_matrix(A_Mat)
        n_cc, labels  = connected_components(sparseA_Mat)
        cc = self.group_nodes_by_component_efficient(labels)
        
        ccImg               = np.copy(img)
        AllClusterPointCloud= []
        crystalAngles       = []
        backboneCoords      = []
        
        # Process each connected component
        for component in cc:
            if len(component) >= self.parameters['cluster threshold size']:
                majorsAxisPointCloud, ellipseAngles = self.process_component(component, polyProps, ccImg)
                AllClusterPointCloud.append(majorsAxisPointCloud)
                crystalAngles.append(np.mean(ellipseAngles))

        # Optionally save backbone coordinates
        self.save_backbone_coords(backboneCoords)
        
        InvCcImg        = invertBinaryImage(ccImg)
        self.debugORSave(img, InvCcImg, 0, "9_CLUSTERS")
        
        return AllClusterPointCloud, crystalAngles
    
    @timeit
    def func_process_cluster(self, ClusterPointCloud, crystalAng, crystal_color):
        processed_clusters = [process_cluster(self.img, cluster, crystalAng[ind], self.parameters, crystal_color) 
                            for ind, cluster in enumerate(ClusterPointCloud)]
        return processed_clusters

    
    @timeit
    def PlottingAndSaving(self, ClusterPointCloud, crystalAng, last_dspace_run = False):
        crystal_color = self.parameters['crystal color']
        processed_clusters = self.func_process_cluster(ClusterPointCloud, crystalAng, crystal_color)
        
        # RGBImg = create_rgb_image(origImg)
        RGBImg = load_img_result_dir(self.img_path, self.parameters)
        
        figure, axes, orig_img_plt_idx, result_img_plt_idx = initialize_plot(last_dspace_run, RGBImg.shape)
        plot_results(last_dspace_run, axes, processed_clusters, self.img, RGBImg, orig_img_plt_idx, result_img_plt_idx)
        
        figure.savefig( join(self.parameters['result image directory'], self.img_path.stem + self.parameters['save image format']))

        if self.parameters['show final image'] == 1:
            plt.show()
        plt.close(figure)
        gc.collect()
        figure.clf()
        return extract_results(processed_clusters)
    
    def process_and_save_dataframe(self, centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference):
    
        if len(centroid) == 0:
            imgNamelist = [self.img_path.name]
        else:
            imgNamelist = [None] * len(centroid)
            imgNamelist[0] = self.img_path.name

        df = pd.DataFrame(list(zip(imgNamelist, centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference)), columns=['Image Name', 'Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)', 'crystalMajorAxis_length (nm)', 'crystalMinorAxis_length (nm)', 'MajorAxisAngle', 'angleDifference'])
        df.round(2)
        
        csv_file_path = Path(self.parameters['result CSV directory']) / f"{self.img_path.stem}.csv"
        # df.to_csv(csv_file_path)
        if csv_file_path.exists():
            print("Appending to existing CSV file")
            df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            print("Creating new CSV file")
            df.to_csv(csv_file_path, mode='w', index=False)
        return df
    
    def process_save_backbone_coords(self, df_boundBox):
    
        if self.parameters['save bounding box'] == 1:
            df_BB = pd.DataFrame( list( zip( df_boundBox ) ) , columns = [ "Top Left(x_y) Bottom Right(x_y)" ] )
            # df_BB.to_csv(join(self.parameters['result annotation directory'], self.img_path.stem+'.csv'))
            csv_file_path = Path(self.parameters['result annotation directory']) / f"{self.img_path.stem}.csv"
            if csv_file_path.exists():
                df_BB.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df_BB.to_csv(csv_file_path, mode='w', index=False)

    @timeit
    def Closing(self, img):
        '''
        Closing: Removing black spots from white regions. Basically Dilation followed by Erosion.
        '''
        
        input   = img
        kernel  = np.ones( ( self.parameters[ 'closing k size' ] , self.parameters[ 'closing k size' ] ) , np.uint8 )
        output  = cv2.morphologyEx( input , cv2.MORPH_CLOSE , kernel )
        
        self.debugORSave( input , output , 1, "2_CLOSING" )
        return output

    @timeit
    def Opening(self, img):
        '''
        Opening: Removing white spots from black regions. Basically Erosion followed by Dilation.
        '''
        input   = img
        kernel  = np.ones(( self.parameters[ 'opening k size' ] , self.parameters[ 'opening k size' ] ) , np.uint8 )
        output  = cv2.morphologyEx( input , cv2.MORPH_OPEN , kernel )
        
        self.debugORSave( input , output , 1, "3_OPENING" )
        return output 

    @timeit
    def Skeletonize(self, img):
        '''
        Skeletonize Image
        ''' 
        input       = img 
        image       = invert( input / np.max( input ) )
        skeleton    = skeletonize( image )
        output      = ( skeleton / np.max( skeleton ) ) * 255 
        InvOutput   = invertBinaryImage( output )

        self.debugORSave( input , InvOutput , 1,"4_SKELETONIZED" )
        return output
    
    def debugORSave(self, initial, final, concat, text):
        if self.parameters['debug'] == 1:
            final = final.astype( 'uint8' )
            final = cv2.equalizeHist( final )
            save_path = Path(self.parameters['result directory']) / f"{text}_{self.img_path.stem}.png"
            cv2.imwrite(str(save_path), final)
    
