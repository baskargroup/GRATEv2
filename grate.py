from utils import *
from ops import *
import time
from skimage import io
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

class ImageProcessor:
    def __init__(self, img_path, parameters):
        self.parameters = parameters
        self.img_path = img_path
        self.img = io.imread(img_path).astype('float64')
        
    @timeit
    def GRATE(self):
        
        self.save_image_to_result_dir()
            
        thresh = self.BlurThresh()
        
        skeleton = self.process_skeleton(thresh)
        
        skeleton = self.BreakBranches(skeleton)
        
        temp = self.SkeletonSegmentation(skeleton)
        
        Broken_backbone_img = self.Filtered_Uniform_BB( temp)
        
        bb_ellipse1, bb_ellipse_props = self.EllipseConstruction(Broken_backbone_img)
        
        adjacencyMat = self.AdjacencyMat(Broken_backbone_img, bb_ellipse_props)
        
        ellipseCluster, AllClusterPointCloud, crystalAngles = self.ConnecComp(Broken_backbone_img, adjacencyMat, bb_ellipse_props)
        
        crystalArea, centroid, crystalAngles_final, dspaces, df_boundBox, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference = self.PlottingAndSaving( AllClusterPointCloud, crystalAngles)
        
        df = self.process_and_save_dataframe(centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference)
        
        self.process_save_backbone_coords(df_boundBox)
        return df
    
    @timeit
    def save_image_to_result_dir(self):
        """Save image to result directory."""
        
        if (self.parameters['result image directory'] / (self.img_path.stem+'.png')).is_file():
            print("Image already present in the result image directory")
        else:
            print("Saving image as png to the result image directory")
            cv2.imwrite(str(self.parameters['result image directory'] / (self.img_path.stem+'.png')), cv2.cvtColor(self.img.astype('uint8'), cv2.COLOR_GRAY2RGB))
        
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
    def EllipseConstruction(self, img):

        input                       = np.copy(img)
        label_img                   = label(input)
        props                       = regionprops_table(label_img, properties = ( 'centroid' , 'orientation' , 'major_axis_length' , 'minor_axis_length', 'coords' ) )
        props                       = pd.DataFrame( props )
        props['orientation']        = - 90 - props[ 'orientation' ] * 180/np.pi
        props['major_axis_length']  = props[ 'major_axis_length' ] / 2
        props['minor_axis_length']  = props[ 'minor_axis_length' ] / 2
        bb_props                    = pd.DataFrame()
        bb_props                    = props
        
        for ind in range(props.shape[0]):
            if props[ 'minor_axis_length' ][ ind ] < 1:
                bb_props = bb_props.drop([ind])   
            
            elif props[ 'major_axis_length' ][ ind ] / props[ 'minor_axis_length' ][ ind ] > self.parameters[ 'ellipse threshold aspect ratio' ]:
                ellip_temp_img = cv2.ellipse( input , ( int( props[ 'centroid-1' ][ ind ] ) , int( props[ 'centroid-0' ][ ind ] ) ) , \
                                            ( int( props[ 'major_axis_length' ][ ind ] ), int( props[ 'minor_axis_length' ][ ind ] ) ) , \
                                            int( props[ 'orientation' ][ ind ] ) , 0.0 , 360.0 , ( 255 , 0 , 0 ) , 2 );
            else: 
                bb_props = bb_props.drop([ind])
        
        bb_props_np         = bb_props.to_numpy()
        InvEllip_temp_img   = invertBinaryImage(ellip_temp_img)
        
        self.debugORSave( img , InvEllip_temp_img , 0, "7_ELLIPSE INSCRIBED" )
        return ellip_temp_img, bb_props_np
    
    @timeit
    def AdjacencyMat(self, img, bb_props):
        '''
        Creating Adjacency Matrix based on distance between centroid and the orientation angle
        '''
        bb_props_np     = bb_props
        centroid_coord  = bb_props_np[ : , : 2 ]
        tree            = KDTree(centroid_coord, leaf_size=2)
        N               = len(bb_props_np)
        KNN_radius      = 2*self.parameters['ellipse pixel size'] + self.parameters['adjacency threshold distance']
        A_Mat           = np.zeros((N,N))

        for i in range(N):
            ind = tree.query_radius(np.reshape(centroid_coord[i],(1,2)), r=KNN_radius)  
            for j in ind[0]:
                if i >= j: # To avoid double counting
                    continue
                pts1        = majorAxisPoints( bb_props_np[ j ] )
                pts2        = majorAxisPoints( bb_props_np[ i ] )
                l2norm      = minDist( pts1 , pts2 )
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
    
    @timeit
    def ConnecComp(self, img, A_Mat, polyProps):
        '''
        Connected Components
        '''
        sparseA_Mat = sp.csr_matrix(A_Mat)
        n_cc, labels  = connected_components(sparseA_Mat)
        cc = self.group_nodes_by_component_efficient(labels)
        
        ccImg                   = np.copy(img)
        AllClusterPointCloud    = []
        crystalAngles           = []

        """ startAngle = 0
        endAngle = 360
        color = (255, 0, 0)
        thickness = 2 """
        backboneCoords = []
        for i in range(len(cc)):
            if len(cc[i]) >= self.parameters['cluster threshold size']:
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
                    # print(polyProps[j][5].shape)
                    backboneCoords.append(polyProps[j][5])
                    majorsAxisPointCloud.append(temp[0,:])
                    majorsAxisPointCloud.append(temp[2,:])
                    ellipseAngels.append(poly[2])
                AllClusterPointCloud.append(majorsAxisPointCloud)
                crystalAngles.append(mean(ellipseAngels))
        
        if self.parameters['save backbone coords'] == 1:
            filehandler = open(join(self.parameters['result backbone coords'], self.img_path.stem+'.pickle'),"wb")
            pickle.dump(backboneCoords, filehandler)
        InvCcImg        = invertBinaryImage(ccImg)
        self.debugORSave(img, InvCcImg, 0, "9_CLUSTERS")
        
        return ccImg, AllClusterPointCloud, crystalAngles
    
    @timeit
    def PlottingAndSaving(self, ClusterPointCloud, crystalAng):
        # RGBImg = create_rgb_image(origImg)
        RGBImg = load_img_result_dir(self.img_path, self.parameters)
        
        figure, axes, orig_img_plt_idx, result_img_plt_idx = initialize_plot()
        
        color_options = ['b', 'g', 'r', 'c', 'm','y','w']
        
        processed_clusters = [process_cluster(self.img, cluster, crystalAng[ind], self.parameters, random.choice(color_options)) 
                            for ind, cluster in enumerate(ClusterPointCloud)]
        
        plot_results(axes, processed_clusters, self.img, RGBImg, orig_img_plt_idx, result_img_plt_idx)
        
        figure.tight_layout()
        figure.savefig( join(self.parameters['result image directory'], self.img_path.stem +'.png') )

        if self.parameters['show final image'] == 1:
            plt.show()
        plt.close(figure)
        gc.collect()
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
        df.to_csv(csv_file_path)
        return df
    
    def process_save_backbone_coords(self, df_boundBox):
    
        if self.parameters['save bounding box'] == 1:
            df_BB = pd.DataFrame( list( zip( df_boundBox ) ) , columns = [ "Top Left(x_y) Bottom Right(x_y)" ] )
            df_BB.to_csv(join(self.parameters['result annotation directory'], self.img_path.stem+'.csv'))

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
    
