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

if __name__ == "__main__":
    project_fPath = pl.Path(__file__).parent.resolve()
    runDir_rPath = 'DATA/BO/results/ryan_allCombined/version_1/'
    csvFile_fPath = project_fPath / runDir_rPath / 'overall_dspace_1.9.csv'
    plotSave_fPath = project_fPath / runDir_rPath / 'Plots'
    
    # Create plotSave_fPath if it does not exist
    plotSave_fPath.mkdir(parents=True, exist_ok=True)
    
    # fileName = '1p9_inRangedspace.csv'
    # filePath = 'forPaper/' + fileName
    
    df = pd.read_csv(csvFile_fPath)
    
    # get crystalArea column
    # crystalArea = df['CrystalArea']
    crystalArea = df['Crystal Area (nm^2)']
    plotHistWithKde(crystalArea, 'Crystal Area (nm$^2$)', 'histogram_crystalArea', plotSave_fPath)