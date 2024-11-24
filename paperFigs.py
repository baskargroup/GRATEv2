import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pl
import libconf
from utils import filterThreshArea, FilterOut_dspacingOutliers
from scipy.stats import wasserstein_distance

def plotHistWithKde(data, 
                    xLabel, 
                    fileName, 
                    plotSave_fPath, 
                    xScale='log', 
                    binsType='fd', 
                    yScale='linear' , 
                    yLabel='Probability Density',
                    x_upper_bound=None,
                    y_upper_bound=None):
    import seaborn as sns
    # create numpy array
    np_data = np.array(data)
    
    # get bins using np.histogram_bin_edges
    bins = np.histogram_bin_edges(np_data, bins=binsType)
    print("bins:  ", bins)
    
    minVal = np.min(np_data)
    maxVal = np.max(np_data)
    
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size' : 60,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    
    # use seaborn to plot histogram with best fit line
    fig, ax = plt.subplots(figsize=(10, 10) )
    sns.histplot(np_data, bins=bins, ax=ax, color='darkorange', stat='density')
    sns.kdeplot(np_data, ax=ax, color='black', clip = (minVal, maxVal), linewidth=2 )
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    # ax.set_ylabel('Normalized Frequency')
    ax.set_xscale(xScale)
    ax.set_yscale(yScale)
    
    # Set x and y axis upper bounds if provided
    if x_upper_bound is not None:
        ax.set_xlim(right=x_upper_bound)
    if y_upper_bound is not None:
        ax.set_ylim(top=y_upper_bound)
    
    # tight layout
    plt.tight_layout()
    
    # Create a directory inside the plotSave_fPath under the name of the fileName
    plotSave_fPath = plotSave_fPath / fileName
    plotSave_fPath.mkdir(parents=True, exist_ok=True)
     
    # save as png and pgf
    fig.savefig(plotSave_fPath / (fileName + '.png'))
    fig.savefig(plotSave_fPath / (fileName + '.pgf'))
    fig.savefig(plotSave_fPath / (fileName + '.pdf'))
    
    plt.close(fig)

def createDataSufficiencyPlots(df, 
                               col_name,
                               numPlots, 
                               xlabel, 
                               fileName, 
                               plotSave_fPath, 
                               xScale='log',  
                               yScale='linear', 
                               yLabel='Probability Density',
                               binsType='fd',):
    
    dataSuffDir_fPath = plotSave_fPath / 'DataSufficiency'
    dataSuffDir_fPath.mkdir(parents=True, exist_ok=True)
    
    for i in range(numPlots):
        numData = int(df.shape[0] * (i+1) / numPlots)
        data = df[col_name].iloc[:numData]
        plotHistWithKde(data, 
                        xlabel, 
                        fileName + '_{}'.format(int((i+1)*100 / numPlots)), 
                        dataSuffDir_fPath, 
                        xScale=xScale, 
                        binsType=binsType, 
                        yScale=yScale , 
                        yLabel=yLabel, 
                        x_upper_bound=None,
                        y_upper_bound=0.006)
    

if __name__ == "__main__":
    project_fPath       = pl.Path(__file__).parent.resolve()
    runDir_rPath        = 'DATA/BO/results/ryan_allCombined/version_1/'
    origCSVFile_fPath   = project_fPath / runDir_rPath / 'overall_dspace_1.9.csv'
    plotSave_fPath      = project_fPath / runDir_rPath / 'Plots'
    config_fPath        = project_fPath / runDir_rPath / 'config.cfg'
    
    # Read config file
    with open(config_fPath, 'r') as f:
        config = libconf.load(f)
        if 'post_processing' not in config:
            raise ValueError("post_processing section not found in config file")
    
    ds_nm                    = config['dspace_nm'][0]
    pp_ds_lowerbound         = config['post_processing']['ds_lower_bound']
    pp_ds_upperbound         = config['post_processing']['ds_upper_bound']
    pp_threshold_area_factor = config['post_processing']['threshold_area_factor']
    
    filteredCSVFile_fPath = project_fPath / runDir_rPath / 'ds_area_filtered_overall_dspace_{}.csv'.format(ds_nm)
    
    # Create plotSave_fPath if it does not exist
    plotSave_fPath.mkdir(parents=True, exist_ok=True)
    
    # Filter out dspacing outliers if filteredCSVFile_fPath does not exist
    if not filteredCSVFile_fPath.exists():
        df = pd.read_csv(origCSVFile_fPath)
        df = FilterOut_dspacingOutliers(df, 
                                        'D-Spacing(FFT, nm)', 
                                        pp_ds_lowerbound, 
                                        pp_ds_upperbound)
        
        # Filter out crystal area outliers
        df = filterThreshArea(df, 
                              'Crystal Area (nm^2)', 
                              ds_nm,
                              pp_threshold_area_factor)
        
        df.to_csv(filteredCSVFile_fPath, index=False)
        
    else:
        df = pd.read_csv(filteredCSVFile_fPath)
    
    # Plotting
    plotHistWithKde(df['Crystal Area (nm^2)'], 
                    'Crystal Area (nm$^2$)', 
                    'histogram_crystalArea', 
                    plotSave_fPath, 
                    xScale='log')
    plotHistWithKde(df['D-Spacing(FFT, nm)'], 
                    'd-spacing (nm)', 
                    'histogram_dspacing', 
                    plotSave_fPath, 
                    xScale='linear')
    plotHistWithKde(df['angleDifference'], 
                    'Angle Difference (degrees)', 
                    'histogram_angleDifference', 
                    plotSave_fPath, 
                    xScale='linear')
    
    aspectRatio = df['crystalMajorAxis_length (nm)'] / df['crystalMinorAxis_length (nm)']
    plotHistWithKde(aspectRatio, 
                    'Aspect Ratio', 
                    'histogram_aspectRatio', 
                    plotSave_fPath, 
                    xScale='linear')
    
    createDataSufficiencyPlots(df, 
                               'Crystal Area (nm^2)', 
                               10, 
                               'Area (nm$^2$)', 
                               'dataSuff_crysArea', 
                               plotSave_fPath, 
                               xScale='log')