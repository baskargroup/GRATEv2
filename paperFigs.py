import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotHistWithKde(data, xLabel, fileName, xScale='log', binsType='fd', yScale='linear' , yLabel='Probability Density'):
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
     
    # save as png and pgf
    fig.savefig(fileName + '.png')
    fig.savefig(fileName + '.pgf')
    fig.savefig(fileName + '.pdf')

if __name__ == "__main__":
    fileName = 'overall_areaFiltered.csv'
    # fileName = '1p9_inRangedspace.csv'
    filePath = 'forPaper/' + fileName
    
    df = pd.read_csv(filePath)
    
    # get crystalArea column
    crystalArea = df['CrystalArea']
    plotHistWithKde(crystalArea, 'Crystal Area (nm$^2$)', 'histogram_crystalArea')