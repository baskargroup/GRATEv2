import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pl

def plotHistWithKde(data, 
                    xLabel, 
                    fileName, 
                    plotSave_fPath, 
                    xScale='log', 
                    binsType='fd', 
                    yScale='linear' , 
                    yLabel='Probability Density'):
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
    
    # tight layout
    plt.tight_layout()
    
    # Create a directory inside the plotSave_fPath under the name of the fileName
    plotSave_fPath = plotSave_fPath / fileName
    plotSave_fPath.mkdir(parents=True, exist_ok=True)
     
    # save as png and pgf
    fig.savefig(plotSave_fPath / (fileName + '.png'))
    fig.savefig(plotSave_fPath / (fileName + '.pgf'))
    fig.savefig(plotSave_fPath / (fileName + '.pdf'))
    
def FilterOut_dspacingOutliers(df, dspaceColName, ds_lowerbound, ds_upperbound, saveFile_fPath):
    # Filter out dspacing outliers
    df_filtered = df[(df[dspaceColName] >= ds_lowerbound) & (df[dspaceColName] <= ds_upperbound)]
    df_filtered.to_csv(saveFile_fPath, index=False)
    return df_filtered

if __name__ == "__main__":
    project_fPath       = pl.Path(__file__).parent.resolve()
    runDir_rPath        = 'DATA/BO/results/ryan_allCombined/version_1/'
    origCSVFile_fPath   = project_fPath / runDir_rPath / 'overall_dspace_1.9.csv'
    plotSave_fPath      = project_fPath / runDir_rPath / 'Plots'
    ds_lowerbound       = 1.5
    ds_upperbound       = 2.8
    
    filteredCSVFile_fPath = project_fPath / runDir_rPath / 'filtered_overall_dspace_1.9.csv'
    
    # Create plotSave_fPath if it does not exist
    plotSave_fPath.mkdir(parents=True, exist_ok=True)
    
    # Filter out dspacing outliers if filteredCSVFile_fPath does not exist
    if not filteredCSVFile_fPath.exists():
        df = pd.read_csv(origCSVFile_fPath)
        df = FilterOut_dspacingOutliers(df, 'D-Spacing(FFT, nm)', ds_lowerbound, ds_upperbound, filteredCSVFile_fPath)
        
        # Store dspacing bounds in a file for future reference inside the runDir
        with open(project_fPath / runDir_rPath / 'filtered_dspacingBounds.txt', 'w') as f:
            f.write(f"ds_lowerbound: {ds_lowerbound}\n")
            f.write(f"ds_upperbound: {ds_upperbound}\n")
        
    else:
        df = pd.read_csv(filteredCSVFile_fPath)
    
    # Plotting
    plotHistWithKde(df['Crystal Area (nm^2)'], 'Crystal Area (nm$^2$)', 'histogram_crystalArea', plotSave_fPath, xScale='log')
    plotHistWithKde(df['D-Spacing(FFT, nm)'], 'd-spacing (nm)', 'histogram_dspacing', plotSave_fPath, xScale='linear')
    plotHistWithKde(df['angleDifference'], 'Angle Difference (degrees)', 'histogram_angleDifference', plotSave_fPath, xScale='linear')
    
    aspectRatio = df['crystalMajorAxis_length (nm)'] / df['crystalMinorAxis_length (nm)']
    plotHistWithKde(aspectRatio, 'Aspect Ratio', 'histogram_aspectRatio', plotSave_fPath, xScale='linear')