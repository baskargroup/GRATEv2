import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from mpl_toolkits.mplot3d import Axes3D
import pathlib as pl
import libconf
from scipy.stats import gaussian_kde, wasserstein_distance

def plot2DKde(data_x, data_y, 
              xLabel, yLabel, 
              fileName, 
              plotSave_fPath, 
              xScale='linear', 
              yScale='linear',
              cmap='viridis'):

    # Ensure data is numpy arrays
    np_data_x = np.array(data_x)
    np_data_y = np.array(data_y)
    
    # Get min and max for axes limits
    min_x, max_x = np.min(np_data_x), np.max(np_data_x)
    min_y, max_y = np.min(np_data_y), np.max(np_data_y)
    
    # Configure matplotlib to use 'pgf' backend for LaTeX integration
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 60,
        'text.usetex': True,
        'pgf.rcfonts': False,
        })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the 2D KDE and store the returned QuadContourSet
    kde = sns.kdeplot(
        x=np_data_x, 
        y=np_data_y, 
        ax=ax, 
        fill=True, 
        cmap=cmap, 
        thresh=0, 
        levels=100,
        cbar=False,          # Enable color bar
        cbar_kws={'label': 'Density'}  # Label for the color bar
        )
    
    # Set labels and scales
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xscale(xScale)
    ax.set_yscale(yScale)
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Increase the number of ticks on x and y axes
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    
    # Optionally rotate tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Create the color bar manually
    # Get the mappable object (the last collection from the plot)
    mappable = kde.collections[-1]
    # Create the color bar
    cbar = fig.colorbar(mappable, 
                        ax=ax, 
                        label='Probability Density', 
                        orientation='horizontal', 
                        pad=0.2, 
                        fraction=0.05, 
                        aspect=30)
    # Adjust the color bar ticks to 4 decimal places
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    
    cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    # Adjust layout to accommodate color bar
    plt.subplots_adjust(bottom=0.2)
    
    # Tight layout
    plt.tight_layout()
    
    # Create the directory to save the plot
    plotSave_fPath = plotSave_fPath / fileName
    plotSave_fPath.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    fig.savefig(plotSave_fPath / f"{fileName}.png")
    fig.savefig(plotSave_fPath / f"{fileName}.pgf")
    fig.savefig(plotSave_fPath / f"{fileName}.pdf")
    
    # Close the figure to free memory
    plt.close(fig)

def plot3DSeabornKde(data_x, data_y,
                     xLabel, yLabel,
                     fileName,
                     plotSave_fPath,
                     xScale='linear',
                     yScale='linear',
                     cmap='viridis'):

    zLabel = 'Probability Density'
    # Ensure data is numpy arrays
    np_data_x = np.array(data_x)
    np_data_y = np.array(data_y)

    # Apply Seaborn style
    sns.set(style='whitegrid', font_scale=1.5)

    # Perform a 2D KDE
    xy = np.vstack([np_data_x, np_data_y])
    kde = gaussian_kde(xy)

    # Define grid over data range
    x_min, x_max = np_data_x.min(), np_data_x.max()
    y_min, y_max = np_data_y.min(), np_data_y.max()

    # Create grid coordinates
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Evaluate KDE on grid
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords)
    z = z.reshape(x_grid.shape)

    # Configure Matplotlib to use 'pgf' backend for LaTeX integration
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 18,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    # Create the figure and 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with Seaborn's colormap
    cmap = sns.color_palette(cmap, 
                             as_cmap=True)
    surf = ax.plot_surface(x_grid, 
                           y_grid, 
                           z, 
                           cmap=cmap, 
                           edgecolor='none', 
                           alpha=0.8, 
                           antialiased=True)

    # Set labels and scales
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    if xScale == 'log':
        ax.set_xscale('log')
    if yScale == 'log':
        ax.set_yscale('log')

    # Apply Seaborn's aesthetic parameters to 3D plot
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)

    # Add a color bar
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(z)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=225)

    # Tight layout
    plt.tight_layout()

    # Create the directory to save the plot
    plotSave_fPath = plotSave_fPath / fileName
    plotSave_fPath.mkdir(parents=True, exist_ok=True)

    # Save the figure
    fig.savefig(plotSave_fPath / f"{fileName}.png")
    fig.savefig(plotSave_fPath / f"{fileName}.pgf")
    fig.savefig(plotSave_fPath / f"{fileName}.pdf")

    # Close the figure to free memory
    plt.close(fig)
    
def plotHistWithKde(data, 
                    xLabel, 
                    fileName, 
                    plotSave_fPath, 
                    xScale='log', 
                    binsType='fd', 
                    yScale='linear' , 
                    yLabel='Probability Density',
                    x_upper_bound=None,
                    y_upper_bound=None,
                    color='darkorange'):
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
    sns.histplot(np_data, bins=bins, ax=ax, color=color, stat='density')
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
                               xlabel, 
                               fileName, 
                               plotSave_fPath,
                               numPlots, 
                               xScale='log',  
                               yScale='linear', 
                               yLabel='Probability Density',
                               binsType='fd',):
    
    dataSuffDir_fPath = plotSave_fPath / 'DataSufficiency'
    dataSuffDir_fPath.mkdir(parents=True, exist_ok=True)
    
    uniformDist =  np.random.uniform(df[col_name].min(), df[col_name].max(), len(df[col_name]))
    
    # Create separate wasserstein distance files to store the wasserstein distance between 
    # 1. the current and previous plot
    # 2. the current and full data 
    # 3. Uniform distribution and the current plot
    wassDist_currPrev_fPath = dataSuffDir_fPath / 'wassDist_currPrev.txt'
    wassDist_currFull_fPath = dataSuffDir_fPath / 'wassDist_currFull.txt'
    wassDist_currUniform_fPath = dataSuffDir_fPath / 'wassDist_currUniform.txt'
    
    wassDist_currPrev_f = open(wassDist_currPrev_fPath, 'w')
    wassDist_currFull_f = open(wassDist_currFull_fPath, 'w')
    wassDist_currUniform_f = open(wassDist_currUniform_fPath, 'w')
    
    wassDist_currPrev_f.write('current data percent, WassDist\n')
    wassDist_currFull_f.write('current data percent, WassDist\n')
    wassDist_currUniform_f.write('current data percent, WassDist\n')
    
    df_current = None
    df_previous = None
    
    for i in range(numPlots):
        numData = int(df.shape[0] * (i+1) / numPlots)
        df_current = df[col_name].iloc[:numData]
        plotHistWithKde(df_current, 
                        xlabel, 
                        fileName + '_{}'.format(int((i+1)*100 / numPlots)), 
                        dataSuffDir_fPath, 
                        xScale=xScale, 
                        binsType=binsType, 
                        yScale=yScale , 
                        yLabel=yLabel, 
                        x_upper_bound=None,
                        y_upper_bound=0.006,
                        color='red')
        
        if df_previous is not None:
            wass_dist = wasserstein_distance(df_previous, df_current)
            wassDist_currPrev_f.write('{}, {}\n'.format(int((i+1)*100 / numPlots), wass_dist))
            # df_previous = df_current
            
        wass_dist = wasserstein_distance(uniformDist[:numData], df_current)
        wassDist_currUniform_f.write('{}, {}\n'.format(int((i+1)*100 / numPlots), wass_dist))
        
        wass_dist = wasserstein_distance(df[col_name], df_current)
        wassDist_currFull_f.write('{}, {}\n'.format(int((i+1)*100 / numPlots), wass_dist))
        
        df_previous = df_current
        
    wassDist_currPrev_f.close()
    wassDist_currFull_f.close()
    wassDist_currUniform_f.close()
    
    
if __name__ == "__main__":
    project_fPath       = pl.Path(__file__).parent.resolve()
    runDir_rPath        = 'DATA/BO/results/ryan_allCombined/version_1/'
    origCSVFile_fPath   = project_fPath / runDir_rPath / 'overall_dspace_1.9.csv'
    plotSave_fPath      = project_fPath / runDir_rPath / 'Plots'
    config_fPath        = project_fPath / runDir_rPath / 'config.cfg'
    csvDir_fPath        = project_fPath / runDir_rPath / 'CSV'
    sameDSCSVFile_fPath = project_fPath / runDir_rPath / 'sameDSpacingInfo.csv'
    filteredOverallCSV_fPath = project_fPath / runDir_rPath / 'ds_area_filtered_overall.csv'
    
    # Read config file
    with open(config_fPath, 'r') as f:
        config = libconf.load(f)
        if 'post_processing' not in config:
            raise ValueError("post_processing section not found in config file")
    
    ds_nm                    = config['dspace_nm'][0]
    pix2nm                   = config['pix_2_nm']
    pp_ds_lowerbound         = config['post_processing']['ds_lower_bound']
    pp_ds_upperbound         = config['post_processing']['ds_upper_bound']
    pp_threshold_area_factor = config['post_processing']['threshold_area_factor']
    
    # Create plotSave_fPath if it does not exist
    plotSave_fPath.mkdir(parents=True, exist_ok=True)
    
    df_filteredOverall = pd.read_csv(filteredOverallCSV_fPath)
    
    if df_filteredOverall.shape[0] == 0:
        raise ValueError("No data to plot")
    
    aspectRatio = df_filteredOverall['crystalMajorAxis_length (nm)'] / df_filteredOverall['crystalMinorAxis_length (nm)']
    
    # Plotting
    plotHistWithKde(df_filteredOverall['Crystal Area (nm^2)'], 
                    'Crystal Area (nm$^2$)', 
                    'histogram_crystalArea', 
                    plotSave_fPath, 
                    xScale='log')
    plotHistWithKde(df_filteredOverall['D-Spacing(FFT, nm)'], 
                    'd-spacing (nm)', 
                    'histogram_dspacing', 
                    plotSave_fPath, 
                    xScale='linear')
    plotHistWithKde(df_filteredOverall['angleDifference'], 
                    'Angle Difference (degrees)', 
                    'histogram_angleDifference', 
                    plotSave_fPath, 
                    xScale='linear')
    plotHistWithKde(aspectRatio, 
                    'Aspect Ratio', 
                    'histogram_aspectRatio', 
                    plotSave_fPath, 
                    xScale='linear')
    
    createDataSufficiencyPlots(df_filteredOverall, 
                               'Crystal Area (nm^2)',  
                               'Area (nm$^2$)', 
                               'dataSuff_crysArea', 
                               plotSave_fPath, 
                               numPlots=10,
                               xScale='log')
    
    df_sameDSCorrel = pd.read_csv(sameDSCSVFile_fPath)
    
    if df_sameDSCorrel.shape[0] == 0:
        raise ValueError("No data to plot")
    
    plotHistWithKde(df_sameDSCorrel['Relative Angle'],
                    'Angle Difference (degrees)',
                    'histogram_angleDifference_correlation',
                    plotSave_fPath,
                    xScale='linear')
    
    plot2DKde(  df_sameDSCorrel['Metric Distances'],
                df_sameDSCorrel['Relative Angle'],
                'Metric Distances',
                'Angle Difference',
                '2D_kde_MetricDist_AngleDiff',
                plotSave_fPath,
                xScale='linear',
                yScale='linear')
    
    plot3DSeabornKde(df_sameDSCorrel['Metric Distances'],
                    df_sameDSCorrel['Relative Angle'],
                    'Metric Distances',
                    'Angle Difference',
                    '3D_kde_MetricDist_AngleDiff',
                    plotSave_fPath,
                    xScale='linear',
                    yScale='linear')