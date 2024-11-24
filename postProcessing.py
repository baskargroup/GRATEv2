import pandas as pd
import pathlib as pl
import libconf
import numpy as np
from utils import filterThreshArea, filterOut_dspacingOutliers, getAngleDifference

def radFromArea(area):
    rad = np.sqrt(area/np.pi)
    return rad

def numericFromString(str, pix2nm):
    str = str[1:-1]
    num = str.split(", ")
    num = list(map(int, num))
    num[0] = num[0]/pix2nm
    num[1] = num[1]/pix2nm
    return num

def centroidDist(cnt1, cnt2):
    np1 = np.array(cnt1)
    np2 = np.array(cnt2)
    return np.linalg.norm(np1-np2)

def create_ds_area_filtered_csv_file(origCSVFile_fPath, 
                                     filteredCSVFile_fPath,
                                     pp_ds_lowerbound,
                                     pp_ds_upperbound,
                                     ds_nm,
                                     pp_threshold_area_factor):
    # Filter out dspacing outliers if filteredCSVFile_fPath does not exist
    df = pd.read_csv(origCSVFile_fPath)
    df = filterOut_dspacingOutliers(df, 
                                    'D-Spacing(FFT, nm)', 
                                    pp_ds_lowerbound, 
                                    pp_ds_upperbound)
    
    # Filter out crystal area outliers
    df = filterThreshArea(  df, 
                            'Crystal Area (nm^2)', 
                            ds_nm,
                            pp_threshold_area_factor)
    
    df = df.round(2)
    df.to_csv(filteredCSVFile_fPath, index=False)
    

def create_same_ds_info_csv(csvDir_fPath, 
                            sameDSCSVFile_fPath,
                            pp_ds_lowerbound,
                            pp_ds_upperbound,
                            ds_nm,
                            pix2nm,
                            pp_threshold_area_factor):
    
    files1 = [f for f in csvDir_fPath.iterdir() if f.suffix == ".csv"]
    print("len files1:  ",len(files1))
    
    MetricDistances = []
    DirectDistances = []
    ModRelAngle     = []

    for filename in files1:
        if filename.stem == "overall":
            continue
        
        print("File name:   ", filename.stem)
        df1 = pd.read_csv(filename)
        df1 = filterOut_dspacingOutliers(df1, 
                                        'D-Spacing(FFT, nm)', 
                                        pp_ds_lowerbound, 
                                        pp_ds_upperbound)
        df1 = filterThreshArea( df1, 
                                'Crystal Area (nm^2)',
                                ds_nm,
                                pp_threshold_area_factor)

        for ind1, row1 in df1.iterrows():
            ang1        = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])
            area1       = float(row1['Crystal Area (nm^2)'])
            rad1        = radFromArea(area1)
            centroid1   = numericFromString(row1['Centroid'], pix2nm)

            for ind2, row2 in df1.iterrows():
                if ind1 == ind2:
                    continue
                else:
                    ang2        = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])
                    area2       = float(row2['Crystal Area (nm^2)'])
                    rad2        = radFromArea(area2)
                    centroid2   = numericFromString(row2['Centroid'], pix2nm)
                    CCdist      = centroidDist(centroid1, centroid2)
                    MetricDist  = CCdist/ (rad1 + rad2)
                    
                    MetricDistances.append(MetricDist)
                    DirectDistances.append(CCdist)
                    ModRelAngle.append(getAngleDifference(ang1, ang2))
                
    print("length:   ", len(ModRelAngle))

    df_dist = pd.DataFrame(list(zip(MetricDistances, 
                                    DirectDistances, 
                                    ModRelAngle)), 
                        columns=['Metric Distances',
                                'Direct Distances',
                                'Relative Angle'])
    df_dist = df_dist.round(2)
    df_dist.to_csv(sameDSCSVFile_fPath, index=False)

if __name__ == "__main__":
    project_fPath       = pl.Path(__file__).parent.resolve()
    runDir_rPath        = 'DATA/BO/results/ryan_allCombined/version_1/'
    origCSVFile_fPath   = project_fPath / runDir_rPath / 'overall_dspace_1.9.csv'
    config_fPath        = project_fPath / runDir_rPath / 'config.cfg'
    csvDir_fPath        = project_fPath / runDir_rPath / 'CSV'
    filteredCSVFile_fPath = project_fPath / runDir_rPath / 'ds_area_filtered_overall.csv'
    sameDSCSVFile_fPath = project_fPath / runDir_rPath / 'sameDSpacingInfo.csv'
    
    # Read config file
    with open(config_fPath, 'r') as f:
        config = libconf.load(f)
        if 'post_processing' not in config:
            raise ValueError("post_processing section not found in config file")
    
    if len(config['dspace_nm']) > 1:
        raise ValueError("Only one dspace_nm value is allowed, found {}".format(len(config['dspace_nm'])))
    
    ds_nm                    = config['dspace_nm'][0]
    pix2nm                   = config['pix_2_nm']
    pp_ds_lowerbound         = config['post_processing']['ds_lower_bound']
    pp_ds_upperbound         = config['post_processing']['ds_upper_bound']
    pp_threshold_area_factor = config['post_processing']['threshold_area_factor']
        
    create_ds_area_filtered_csv_file(origCSVFile_fPath,
                                     filteredCSVFile_fPath,
                                     pp_ds_lowerbound,
                                     pp_ds_upperbound,
                                     ds_nm,
                                     pp_threshold_area_factor)
    
    create_same_ds_info_csv(csvDir_fPath,
                            sameDSCSVFile_fPath,
                            pp_ds_lowerbound,
                            pp_ds_upperbound,
                            ds_nm,
                            pix2nm,
                            pp_threshold_area_factor)
        
    