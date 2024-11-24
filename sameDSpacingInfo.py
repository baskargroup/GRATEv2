import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import os
import libconf
from os import listdir
from os.path import join, splitext
from utils import plotHist, filterThreshArea, getAngleDifference, FilterOut_dspacingOutliers
from paperFigs import plotHistWithKde

def listIntersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

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

def radFromArea(area):
    rad = np.sqrt(area/np.pi)
    return rad
    
def plot2DKde(data_x, data_y, 
              xLabel, yLabel, 
              fileName, 
              plotSave_fPath, 
              xScale='linear', 
              yScale='linear',
              cmap='viridis'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from pathlib import Path
    import matplotlib.ticker as ticker

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
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from pathlib import Path
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import gaussian_kde

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
    cmap = sns.color_palette(cmap, as_cmap=True)
    surf = ax.plot_surface(x_grid, y_grid, z, cmap=cmap, edgecolor='none', alpha=0.8, antialiased=True)

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
    plotSave_fPath = Path(plotSave_fPath) / fileName
    plotSave_fPath.mkdir(parents=True, exist_ok=True)

    # Save the figure
    fig.savefig(plotSave_fPath / f"{fileName}.png")
    fig.savefig(plotSave_fPath / f"{fileName}.pgf")
    fig.savefig(plotSave_fPath / f"{fileName}.pdf")

    # Close the figure to free memory
    plt.close(fig)


if __name__ == "__main__":
    
    project_fPath   = pl.Path(__file__).parent.resolve()
    runDir_rPath    = 'DATA/BO/results/ryan_allCombined/version_1'           ## Set this value
    origCSVFile_fPath   = project_fPath / runDir_rPath / 'overall_dspace_1.9.csv'
    plotSave_fPath      = project_fPath / runDir_rPath / 'Plots'
    config_fPath        = project_fPath / runDir_rPath / 'config.cfg'
    d_spaceDir1         = project_fPath / runDir_rPath / 'CSV'
    sameDSCSVFile_fPath = project_fPath / runDir_rPath / 'sameDSpacingInfo.csv'

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

    plotSave_fPath.mkdir(parents=True, exist_ok=True)

    files1 = [f for f in listdir(d_spaceDir1) if splitext(f)[1] == ".csv"]

    print("len files1:  ",len(files1))

    distances = []

    MetricDistances = []
    DirectDistances = []
    ModRelAngle     = []

    for filename in files1:
        if filename == "overall.csv":
            continue
        print("File name:   ", filename)
        df1 = pd.read_csv(join(d_spaceDir1, filename))
        df1 = FilterOut_dspacingOutliers(df1, 
                                        'D-Spacing(FFT, nm)', 
                                        pp_ds_lowerbound, 
                                        pp_ds_upperbound)
        df1 = filterThreshArea( df1, 
                                'Crystal Area (nm^2)',
                                ds_nm,
                                pp_threshold_area_factor)

        for ind1, row1 in df1.iterrows():
            # print("ind1     :", ind1)
            ang1        = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])
            area1       = float(row1['Crystal Area (nm^2)'])
            rad1        = radFromArea(area1)
            centroid1   = numericFromString(row1['Centroid'], pix2nm)

            for ind2, row2 in df1.iterrows():
                # print("ind2     :", ind2)
                if ind1 == ind2:
                    continue
                else:
                    ang2        = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])
                    area2       = float(row2['Crystal Area (nm^2)'])
                    rad2        = radFromArea(area2)
                    centroid2   = numericFromString(row2['Centroid'], pix2nm)
                    
                    CCdist      = centroidDist(centroid1, centroid2)
                    MetricDist   = CCdist/ (rad1 + rad2)
                    
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
    
    plotHistWithKde(df_dist['Relative Angle'],
                    'Angle Difference (degrees)',
                    'histogram_angleDifference_correlation',
                    plotSave_fPath,
                    xScale='linear')
    
    plot2DKde(df_dist['Metric Distances'],
                df_dist['Relative Angle'],
                'Metric Distances',
                'Angle Difference',
                '2D_kde_MetricDist_AngleDiff',
                plotSave_fPath,
                xScale='linear',
                yScale='linear')
    
    plot3DSeabornKde(df_dist['Metric Distances'],
                    df_dist['Relative Angle'],
                    'Metric Distances',
                    'Angle Difference',
                    '3D_kde_MetricDist_AngleDiff',
                    plotSave_fPath,
                    xScale='linear',
                    yScale='linear')