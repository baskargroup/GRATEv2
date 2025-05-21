import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, splitext
from utils import getAngleDifference, filterThreshArea
from scipy.stats import poisson, gaussian_kde

#############################
#       CONFIGURATION       #
#############################

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = join(PROJECT_PATH, 'Results/ryan/original_analysis/Results_v3')
D_SPACE_DIR_1 = join(DATA_DIR, "1p9/CSV")   # 1.9nm directory
D_SPACE_DIR_2 = join(DATA_DIR, "0p7/CSV")   # 0.7nm directory
PIX2NM = 78.5

#############################
#       UTILITIES           #
#############################

def listIntersection(lst1, lst2):
    """Return a list of items common to both input lists."""
    return [value for value in lst1 if value in lst2]

def numericFromString(s, pix2nm):
    """
    Convert a string representing a tuple (e.g., "(123, 456)") into a list of two numbers scaled by pix2nm.
    """
    s = s[1:-1]
    num = s.split(", ")
    num = list(map(int, num))
    return [num[0] / pix2nm, num[1] / pix2nm]

def centroidDist(cnt1, cnt2):
    """Compute Euclidean distance between two centroids."""
    return np.linalg.norm(np.array(cnt1) - np.array(cnt2))

def radFromArea(area):
    """Return the radius corresponding to an area (assuming a circular shape)."""
    return np.sqrt(area / np.pi)

def freedman_diaconis_bins(data):
    """
    Calculate the optimal number of bins using the Freedman-Diaconis rule.
    Any NaN values in the data are removed before calculation.
    
    Parameters:
      data: 1D array-like numerical data.
      
    Returns:
      bins: The computed number of bins (an integer).
    """
    # Remove NaN values
    if isinstance(data, pd.Series):
        data = data.dropna().values
    else:
        data = np.array(data)
        data = data[~np.isnan(data)]
    
    if len(data) < 2:
        return 1  # Not enough data.
    
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return int(np.ceil(np.sqrt(len(data))))
    
    bin_width = 2 * iqr / np.power(len(data), 1/3)
    data_range = data.max() - data.min()
    if bin_width == 0 or data_range == 0:
        return int(np.ceil(np.sqrt(len(data))))
    
    return int(np.ceil(data_range / bin_width))
  
def order_by_image_crystal_count(df, image_col="ImageName"):
    """
    Order the rows of a DataFrame based on the number of crystal pairs per image.
    Images with the highest number of crystal pairs are placed first.
    
    Parameters:
      df: DataFrame containing an 'ImageName' column.
      image_col: The name of the image column (default "ImageName").
      
    Returns:
      A DataFrame sorted by descending crystal count per image.
    """
    counts = df.groupby(image_col).size().sort_values(ascending=False)
    ordered_images = counts.index.tolist()
    # Create a categorical type with the ordered images and sort the dataframe accordingly.
    df[image_col] = pd.Categorical(df[image_col], categories=ordered_images, ordered=True)
    return df.sort_values(image_col).reset_index(drop=True)

#############################
#   CSV CREATION FUNCTIONS  #
#############################

def createAcrossDSpacingInfo_csv():
    """
    Process CSV files from both directories and create a DataFrame containing:
      - ImageName: Name of the source file (image).
      - Metric Distances: Normalized distance between crystals.
      - Direct Distances: Euclidean distance (in nm) between crystal centroids.
      - Relative Angle: Absolute difference between crystal orientations.
    
    Returns:
      DataFrame with the processed data.
    """
    files1 = [f for f in listdir(D_SPACE_DIR_1) if splitext(f)[1] == ".csv"]
    files2 = [f for f in listdir(D_SPACE_DIR_2) if splitext(f)[1] == ".csv"]
    commonCSV = listIntersection(files1, files2)
    
    print("Files in Dir1:", len(files1))
    print("Files in Dir2:", len(files2))
    print("Common CSV files:", len(commonCSV))
    
    MetricDistances = []
    DirectDistances = []
    ModRelAngle = []
    ImageNames = []  # Record the image filename for each pair
    
    for filename in commonCSV:
        if filename == "overall.csv":
            continue
        df1 = pd.read_csv(join(D_SPACE_DIR_1, filename))
        df1 = filterThreshArea(df1, 'Crystal Area (nm^2)', d_space=1.9, threshold_area_factor=7)
        df2 = pd.read_csv(join(D_SPACE_DIR_2, filename))
        df2 = filterThreshArea(df2, 'Crystal Area (nm^2)', d_space=0.7, threshold_area_factor=10)
        
        for ind1, row1 in df1.iterrows():
            centroid1 = numericFromString(row1['Centroid'], PIX2NM)
            area1 = float(row1['Crystal Area (nm^2)'])
            rad1 = radFromArea(area1)
            ang1 = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])
            for ind2, row2 in df2.iterrows():
                centroid2 = numericFromString(row2['Centroid'], PIX2NM)
                area2 = float(row2['Crystal Area (nm^2)'])
                rad2 = radFromArea(area2)
                ang2 = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])
                CCdist = centroidDist(centroid1, centroid2)
                MetricDist = CCdist / (rad1 + rad2)
                MetricDistances.append(MetricDist)
                DirectDistances.append(CCdist)
                ModRelAngle.append(getAngleDifference(ang1, ang2))
                ImageNames.append(filename)
    
    print("Total pairs processed:", len(DirectDistances))
    df = pd.DataFrame(list(zip(ImageNames, MetricDistances, DirectDistances, ModRelAngle)),
                      columns=['ImageName', 'Metric Distances', 'Direct Distances', 'Relative Angle'])
    return df.round(2)
  
  
def createAcrossDSpacingInfo_with_radii_csv():
    """
    Process CSV files from both directories and create a DataFrame containing:
      - ImageName: Name of the source file (image).
      - Radius1: Radius (nm) of the 0.4 nm crystal.
      - Radius2: Radius (nm) of the 2.0 nm crystal.
      - Metric Distances: Normalized distance between crystals (centroid-to-centroid divided by radii sum).
      - Direct Distances: Euclidean distance (in nm) between crystal centroids.
      - Relative Angle: Absolute difference between crystal orientations (degrees).
    
    Returns:
      DataFrame with the processed data.
    """
    files1 = [f for f in listdir(D_SPACE_DIR_1) if splitext(f)[1] == ".csv"]
    files2 = [f for f in listdir(D_SPACE_DIR_2) if splitext(f)[1] == ".csv"]
    commonCSV = listIntersection(files1, files2)
    
    records = []
    for filename in commonCSV:
        if filename == "overall.csv":
            continue
        df1 = pd.read_csv(join(D_SPACE_DIR_1, filename))
        df1 = filterThreshArea(df1, 'Crystal Area (nm^2)', d_space=1.9, threshold_area_factor=7)
        df2 = pd.read_csv(join(D_SPACE_DIR_2, filename))
        df2 = filterThreshArea(df2, 'Crystal Area (nm^2)', d_space=0.7, threshold_area_factor=10)
        
        for _, row1 in df1.iterrows():
            centroid1 = numericFromString(row1['Centroid'], PIX2NM)
            area1     = float(row1['Crystal Area (nm^2)'])
            rad1      = radFromArea(area1)
            ang1      = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])
            for _, row2 in df2.iterrows():
                centroid2 = numericFromString(row2['Centroid'], PIX2NM)
                area2     = float(row2['Crystal Area (nm^2)'])
                rad2      = radFromArea(area2)
                ang2      = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])
                
                cc_dist   = centroidDist(centroid1, centroid2)
                metric    = cc_dist / (rad1 + rad2)
                rel_ang   = getAngleDifference(ang1, ang2)
                
                records.append({
                    'ImageName':        filename,
                    'Radius1':          rad1,
                    'Radius2':          rad2,
                    'Metric Distances': metric,
                    'Direct Distances': cc_dist,
                    'Relative Angle':   rel_ang
                })
    
    df = pd.DataFrame(records)
    return df.round(2)

def createAvgAngle_acrossDSpacingInfo_csv():
    """
    Process CSV files from both directories and create a DataFrame containing:
      - Avg Relative Angle (difference between the average angles of 0.4nm and 2.0nm crystals)
      - Standard Deviation of Angle for 1.9nm crystals
      - Standard Deviation of Angle for 0.7nm crystals
    
    Returns:
      DataFrame with the processed average and standard deviation data.
    """
    files1 = [f for f in listdir(D_SPACE_DIR_1) if splitext(f)[1] == ".csv"]
    files2 = [f for f in listdir(D_SPACE_DIR_2) if splitext(f)[1] == ".csv"]
    commonCSV = listIntersection(files1, files2)
    
    print("Files in Dir1:", len(files1))
    print("Files in Dir2:", len(files2))
    print("Common CSV files:", len(commonCSV))
    
    ModAvgRelAngle = []
    ModStdRelAngle_1p9 = []
    ModStdRelAngle_0p7 = []
    
    for filename in commonCSV:
        if filename == "overall.csv":
            continue
        df1 = pd.read_csv(join(D_SPACE_DIR_1, filename))
        df1 = filterThreshArea(df1, 'Crystal Area (nm^2)', d_space=1.9, threshold_area_factor=7)
        df2 = pd.read_csv(join(D_SPACE_DIR_2, filename))
        df2 = filterThreshArea(df2, 'Crystal Area (nm^2)', d_space=0.7, threshold_area_factor=10)
        
        df1_ang_avg = df1['Crystal Angle (zero at X-axis and clockwise positive)'].mean()
        df2_ang_avg = df2['Crystal Angle (zero at X-axis and clockwise positive)'].mean()
        df1_ang_std = df1['Crystal Angle (zero at X-axis and clockwise positive)'].std()
        df2_ang_std = df2['Crystal Angle (zero at X-axis and clockwise positive)'].std()
        
        ModAvgRelAngle.append(getAngleDifference(df1_ang_avg, df2_ang_avg))
        ModStdRelAngle_1p9.append(df1_ang_std)
        ModStdRelAngle_0p7.append(df2_ang_std)
    
    df = pd.DataFrame(list(zip(ModAvgRelAngle, ModStdRelAngle_1p9, ModStdRelAngle_0p7)),
                      columns=['Avg Relative Angle', 'Std Angle 1.9nm', 'Std Angle 0.7nm'])
    return df.round(2)

#############################
#       FILTERING           #
#############################

def filter_images_and_crystals(csv_filepath, 
                               peak_direct_range=(10,15), peak_angle_range=(5,10),
                               away_direct_thresh=35, away_angle_thresh=50):
    """
    Read the acrossDSpacingInfo CSV file and filter records into two groups:
      1. Peak Region: Direct Distances between peak_direct_range and Relative Angle within peak_angle_range.
      2. Away Region: Direct Distances greater than away_direct_thresh and Relative Angle greater than away_angle_thresh.
    
    Parameters:
      csv_filepath: Path to the acrossDSpacingInfo CSV file.
      peak_direct_range: Tuple specifying the range of Direct Distances for the peak region.
      peak_angle_range: Tuple specifying the range of Relative Angles for the peak region.
      away_direct_thresh: Lower threshold for Direct Distances in the away region.
      away_angle_thresh: Lower threshold for Relative Angles in the away region.
    
    Returns:
      peak_df: DataFrame containing records in the peak region.
      away_df: DataFrame containing records in the away region.
    """
    df = pd.read_csv(csv_filepath)
    
    # Filter for peak region
    peak_df = df[(df['Direct Distances'] >= peak_direct_range[0]) & 
                 (df['Direct Distances'] <= peak_direct_range[1]) & 
                 (df['Relative Angle'] >= peak_angle_range[0]) & 
                 (df['Relative Angle'] <= peak_angle_range[1])]
    
    # Filter for away region
    away_df = df[(df['Direct Distances'] > away_direct_thresh) & 
                 (df['Relative Angle'] > away_angle_thresh)]
    
    # Order the rows by the number of crystal pairs per image (highest first)
    peak_df = order_by_image_crystal_count(peak_df, "ImageName")
    away_df = order_by_image_crystal_count(away_df, "ImageName")
    
    return peak_df, away_df

#############################
#       PLOTTING            #
#############################

def plot_histogram_avg_relative_angle(df, save_path):
    """
    Plot histogram of 'Avg Relative Angle' using Freedman-Diaconis rule for binning.
    
    Parameters:
      df: DataFrame containing 'Avg Relative Angle'
      save_path: File path to save the plot image.
    """
    bins_avg_rel = freedman_diaconis_bins(df['Avg Relative Angle'])
    plt.figure(figsize=(10, 5))
    plt.hist(df['Avg Relative Angle'], bins=bins_avg_rel, alpha=0.7, label='Avg Relative Angle')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Average Relative Angles')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_histogram_std_angles(df, save_path):
    """
    Plot overlaid histograms of 'Std Angle 1.9nm' and 'Std Angle 0.7nm' using common binning.
    
    Parameters:
      df: DataFrame containing the standard deviation columns.
      save_path: File path to save the plot image.
    """
    combined_std = pd.concat([df['Std Angle 1.9nm'], df['Std Angle 0.7nm']])
    bins_std = freedman_diaconis_bins(combined_std)
    plt.figure(figsize=(10, 5))
    plt.hist(df['Std Angle 1.9nm'], bins=bins_std, alpha=0.7, label='Std Angle 1.9nm')
    plt.hist(df['Std Angle 0.7nm'], bins=bins_std, alpha=0.7, label='Std Angle 0.7nm')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Standard Deviations of Angles')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_histogram_metric_direct(df, save_path):
    """
    Plot overlaid histograms of 'Metric Distances' and 'Direct Distances' using common binning.
    
    Parameters:
      df: DataFrame containing the distance columns.
      save_path: File path to save the plot image.
    """
    combined_dist = pd.concat([df['Metric Distances'], df['Direct Distances']])
    bins_dist = freedman_diaconis_bins(combined_dist)
    plt.figure(figsize=(10, 5))
    plt.hist(df['Metric Distances'], bins=bins_dist, alpha=0.7, label='Metric Distances')
    plt.hist(df['Direct Distances'], bins=bins_dist, alpha=0.7, label='Direct Distances')
    plt.xlabel('Distance (nm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Metric and Direct Distances')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_histogram_relative_angle(df, save_path):
    """
    Plot histogram of 'Relative Angle' using Freedman-Diaconis rule for binning.
    
    Parameters:
      df: DataFrame containing 'Relative Angle'
      save_path: File path to save the plot image.
    """
    bins_rel = freedman_diaconis_bins(df['Relative Angle'])
    plt.figure(figsize=(10, 5))
    plt.hist(df['Relative Angle'], bins=bins_rel, alpha=0.7, label='Relative Angle')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Relative Angles')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

#############################
#    HISTOGRAM PEAK CHECK   #
#############################

def check_histogram_peak(csv_filepath, column='value', num_bins=20, significance_level=0.05):
    """
    Reads a CSV file, computes a histogram for a given column, and determines if the highest peak is statistically significant.
    The significance is evaluated by comparing the observed peak count against the expected count (mean of bin counts)
    assuming a Poisson distribution.
    
    Parameters:
      csv_filepath: Path to the CSV file.
      column: The column name to analyze (default is 'value').
      num_bins: Number of bins for the histogram (default is 20).
      significance_level: p-value threshold for significance (default is 0.05).
    
    Returns:
      is_significant: Boolean indicating if the peak is statistically significant.
      p_value: The computed p-value.
      details: A dictionary with additional details.
    """
    df = pd.read_csv(csv_filepath)
    data = df[column].dropna()
    
    counts, bin_edges = np.histogram(data, bins=num_bins)
    peak_index = np.argmax(counts)
    peak_count = counts[peak_index]
    expected_count = np.mean(counts)
    p_value = poisson.sf(peak_count - 1, expected_count)
    
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvspan(bin_edges[peak_index], bin_edges[peak_index+1], color='red', alpha=0.3, label='Peak bin')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column} with Peak Highlighted')
    plt.legend()
    plt.savefig(csv_filepath.replace('.csv', '_histogram.png'))
    plt.close()
    
    details = {
        'peak_count': peak_count,
        'expected_count': expected_count,
        'p_value': p_value,
        'counts': counts,
        'bin_edges': bin_edges,
        'peak_index': peak_index,
    }
    
    return p_value < significance_level, p_value, details

def check_histogram_peak_FD(csv_filepath, column='value', significance_level=0.05):
    """
    Reads a CSV file, computes a histogram for a given column using the Freedman-Diaconis rule,
    and determines if the highest peak is statistically significant.
    
    Parameters:
      csv_filepath: Path to the CSV file.
      column: The column name to analyze (default is 'value').
      significance_level: p-value threshold for significance (default is 0.05).
    
    Returns:
      is_significant: Boolean indicating if the peak is statistically significant.
      p_value: The computed p-value.
      details: A dictionary with additional details.
    """
    df = pd.read_csv(csv_filepath)
    data = df[column].dropna()
    
    num_bins = freedman_diaconis_bins(data)
    print(f"Using {num_bins} bins (Freedman-Diaconis rule) for the histogram.")
    
    counts, bin_edges = np.histogram(data, bins=num_bins)
    peak_index = np.argmax(counts)
    peak_count = counts[peak_index]
    expected_count = np.mean(counts)
    p_value = poisson.sf(peak_count - 1, expected_count)
    
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvspan(bin_edges[peak_index], bin_edges[peak_index+1], color='red', alpha=0.3, label='Peak bin')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column} with Peak Highlighted')
    plt.legend()
    plt.show()
    
    details = {
        'peak_count': peak_count,
        'expected_count': expected_count,
        'p_value': p_value,
        'counts': counts,
        'bin_edges': bin_edges,
        'peak_index': peak_index,
    }
    
    return p_value < significance_level, p_value, details
  
def plot_contour_direct_angle(csv_filepath, save_path, num_levels=10):
  """
  Reads the acrossDSpacingInfo.csv file, extracts the 'Direct Distances' and 'Relative Angle' columns,
  computes 2D binning using the Freedman-Diaconis rule, and produces a 2D contour plot.
  
  Parameters:
    csv_filepath: Path to the acrossDSpacingInfo.csv file.
    save_path: Path to save the generated contour plot image.
  """
  # Read the CSV file
  df = pd.read_csv(csv_filepath)
  
  # Extract the two columns of interest, dropping NaNs
  direct = df['Direct Distances'].dropna().values
  angle = df['Relative Angle'].dropna().values
  
  # Determine the optimal number of bins for each dimension using the Freedman-Diaconis rule
  num_bins_direct = freedman_diaconis_bins(direct)
  num_bins_angle = freedman_diaconis_bins(angle)
  
  # Compute a 2D histogram of the data
  H, xedges, yedges = np.histogram2d(direct, angle, bins=[num_bins_direct, num_bins_angle])
  
  # Calculate bin centers for both dimensions
  xcenters = (xedges[:-1] + xedges[1:]) / 2
  ycenters = (yedges[:-1] + yedges[1:]) / 2
  X, Y = np.meshgrid(xcenters, ycenters)
  
  # Create a 2D contour plot
  plt.figure(figsize=(10, 8))
  contour = plt.contourf(X, Y, H.T, levels=num_levels,cmap='viridis')
  plt.xlabel('Direct Distances (nm)')
  plt.ylabel('Relative Angle (degrees)')
  plt.title('2D Contour Plot: Direct Distance vs. Relative Angle')
  plt.colorbar(contour, label='Counts')
  plt.savefig(save_path)
  plt.close()


# Kernel Density Estimate (KDE) Plotting Functions

def plot_kde_avg_relative_angle(df, save_path):
    """
    Plot a kernel density estimate (KDE) for 'Avg Relative Angle'.
    
    Parameters:
      df: DataFrame containing 'Avg Relative Angle'.
      save_path: File path to save the KDE plot image.
    """
    data = df['Avg Relative Angle'].dropna().values
    density = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 200)
    ys = density(xs)
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, color='blue', lw=2, label='KDE')
    plt.xlabel('Avg Relative Angle (degrees)')
    plt.ylabel('Density')
    plt.title('KDE of Average Relative Angles')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_kde_std_angles(df, save_path):
    """
    Plot overlaid KDEs for 'Std Angle 1.9nm' and 'Std Angle 0.7nm'.
    
    Parameters:
      df: DataFrame containing the standard deviation columns.
      save_path: File path to save the KDE plot image.
    """
    data1 = df['Std Angle 1.9nm'].dropna().values
    data2 = df['Std Angle 0.7nm'].dropna().values
    density1 = gaussian_kde(data1)
    density2 = gaussian_kde(data2)
    xs1 = np.linspace(data1.min(), data1.max(), 200)
    xs2 = np.linspace(data2.min(), data2.max(), 200)
    plt.figure(figsize=(10, 5))
    plt.plot(xs1, density1(xs1), label='Std Angle 1.9nm', lw=2)
    plt.plot(xs2, density2(xs2), label='Std Angle 0.7nm', lw=2)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Density')
    plt.title('KDE of Standard Deviations of Angles')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_kde_metric_direct(df, save_path):
    """
    Plot overlaid KDEs for 'Metric Distances' and 'Direct Distances'.
    
    Parameters:
      df: DataFrame containing the distance columns.
      save_path: File path to save the KDE plot image.
    """
    data1 = df['Metric Distances'].dropna().values
    data2 = df['Direct Distances'].dropna().values
    density1 = gaussian_kde(data1)
    density2 = gaussian_kde(data2)
    xs1 = np.linspace(data1.min(), data1.max(), 200)
    xs2 = np.linspace(data2.min(), data2.max(), 200)
    plt.figure(figsize=(10, 5))
    plt.plot(xs1, density1(xs1), label='Metric Distances', lw=2)
    plt.plot(xs2, density2(xs2), label='Direct Distances', lw=2)
    plt.xlabel('Distance (nm)')
    plt.ylabel('Density')
    plt.title('KDE of Metric and Direct Distances')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_kde_relative_angle(df, save_path):
    """
    Plot a KDE for 'Relative Angle'.
    
    Parameters:
      df: DataFrame containing 'Relative Angle'.
      save_path: File path to save the KDE plot image.
    """
    data = df['Relative Angle'].dropna().values
    density = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 200)
    ys = density(xs)
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, color='green', lw=2, label='KDE')
    plt.xlabel('Relative Angle (degrees)')
    plt.ylabel('Density')
    plt.title('KDE of Relative Angles')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
    
def plot_wedge_normalized_hist2d(csv_filepath, save_path, num_levels=10):
    """
    Plot a wedge‐normalized 2D histogram of Direct Distance vs Relative Angle.
    Each bin count H(r,θ) is divided by the wedge area r * dr * dθ.
    
    Parameters:
      csv_filepath: path to acrossDSpacingInfo.csv
      save_path: path to save the resulting contour plot
      num_levels: number of contour levels to draw
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 1) load data
    df = pd.read_csv(csv_filepath)
    direct = df['Direct Distances'].dropna().values
    angle  = df['Relative Angle']   .dropna().values

    # 2) choose bin counts via Freedman-Diaconis
    # from your_module import freedman_diaconis_bins
    nbins_r     = freedman_diaconis_bins(direct)
    nbins_theta = freedman_diaconis_bins(angle)

    # 3) compute 2D histogram
    H, r_edges, theta_edges = np.histogram2d(
        direct, angle,
        bins=[nbins_r, nbins_theta]
    )

    # 4) compute bin centers and widths
    r_centers     = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    dr            = np.diff(r_edges)
    dtheta        = np.diff(theta_edges)

    # 5) build grids for normalization
    # R, TH both shape (nbins_r, nbins_theta)
    R, TH = np.meshgrid(r_centers, theta_centers, indexing='ij')
    
    # wedge area = r * dr * dθ
    area = R * dr[:, None] * dtheta[None, :]

    # 6) normalize
    H_norm = H / area

    # 7) contour‐plot
    plt.figure(figsize=(8,6))
    cf = plt.contourf(
        R, TH,          # both (nbins_r, nbins_theta)
        H_norm,         # also (nbins_r, nbins_theta)
        levels=num_levels,
        cmap='viridis'
    )
    plt.xlabel('Direct Distance (nm)')
    plt.ylabel('Relative Angle (°)')
    plt.title('Wedge‐Normalized 2D Histogram')
    plt.colorbar(cf, label='Normalized density')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_kde_contour_direct_angle(
    csv_filepath,
    save_path,
    kde_csv_path=None,
    num_levels=10,
    normalize_by_wedge=True,
    epsilon=1e-3,
    xlim=None,
    ylim=None,
    vmin=None,
    vmax=None,
    log_scale=False
):
    """
    Generate a 2D KDE contour plot for 'Direct Distances' and 'Relative Angle' using Gaussian KDE,
    with optional wedge-area normalization, CSV export, and zoomed axes.

    Parameters:
      csv_filepath: Path to acrossDSpacingInfo.csv.
      save_path: Path to save the KDE contour plot image.
      kde_csv_path: Optional; path to save the KDE grid and density values as a CSV file.
      num_levels: Number of contour levels (default is 10).
      normalize_by_wedge: If True, divide density by radius (xi) to normalize by annular area.
      epsilon: Small value to floor xi and avoid division by zero (default 1e-3 nm).
      xlim: Tuple (xmin, xmax) to set the x-axis limits; if None, uses full range.
      ylim: Tuple (ymin, ymax) to set the y-axis limits; if None, uses full range.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LogNorm

    # 1. Load and clean data
    df = pd.read_csv(csv_filepath)
    direct = df['Direct Distances'].dropna().values
    angle  = df['Relative Angle']   .dropna().values

    # 2. Define uniform grid for KDE evaluation
    x_min, x_max = direct.min(), direct.max()
    y_min, y_max = angle.min(), angle.max()
    xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # 3. Evaluate 2D Gaussian KDE on grid
    values = np.vstack([direct, angle])
    kernel = gaussian_kde(values, bw_method=0.3)
    zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    # 4. Optional wedge-area normalization
    if normalize_by_wedge:
        R = np.maximum(xi, epsilon)
        zi = zi / R

    # 5. Optionally save KDE grid & density to CSV
    if kde_csv_path:
        pd.DataFrame({
            'direct_distance': xi.flatten(),
            'relative_angle':   yi.flatten(),
            'density':          zi.flatten()
        }).to_csv(kde_csv_path, index=False)
        
    # Choose norm
    norm = None
    if log_scale:
        # choose vmin if not set, avoid zeros
        vmin_ = vmin or max(zi[zi>0].min(), 1e-8)
        vmax_ = vmax or zi.max()
        norm = LogNorm(vmin=vmin_, vmax=vmax_)
    else:
        vmin_ = vmin
        vmax_ = vmax

    # 6. Plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf( xi, yi, zi, 
                            levels=num_levels, 
                            cmap='viridis',
                            norm=norm,
                            vmin=vmin_, vmax=vmax_)
    
    plt.xlabel('Direct Distance (nm)')
    plt.ylabel('Relative Angle (°)')
    title = '2D KDE Contour Plot'
    title += ' (Wedge-Normalized)' if normalize_by_wedge else ' (Raw)'
    if log_scale:
        title += ' [Log scale]'
    plt.title(title)
    plt.colorbar(contour, label='Density')

    # 7. Apply axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# def plot_kde_contour_direct_angle(csv_filepath, save_path, 
#                                   kde_csv_path=None, num_levels=10,
#                                   normalize_by_wedge=True,
#                                   epsilon=1e-3):
#     """
#     Generate a 2D KDE contour plot for 'Direct Distances' and 'Relative Angle' using Gaussian KDE,
#     with optional wedge-area normalization and CSV export of the KDE grid.

#     Parameters:
#       csv_filepath: Path to acrossDSpacingInfo.csv.
#       save_path: Path to save the KDE contour plot image.
#       kde_csv_path: Optional; path to save the KDE grid and density values as a CSV file.
#       num_levels: Number of contour levels (default is 10).
#       normalize_by_wedge: If True, divide density by radius (xi) to normalize by annular area.
#       epsilon: Small value to floor xi and avoid division by zero (default 1e-3 nm).
#     """
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import gaussian_kde

#     # Load and clean data
#     df = pd.read_csv(csv_filepath)
#     direct = df['Direct Distances'].dropna().values
#     angle  = df['Relative Angle']   .dropna().values

#     # Define uniform grid for KDE evaluation
#     xmin, xmax = direct.min(), direct.max()
#     ymin, ymax = angle.min(), angle.max()
#     xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

#     # Evaluate 2D Gaussian KDE on grid
#     values = np.vstack([direct, angle])
#     kernel = gaussian_kde(values, bw_method=0.3)
#     zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
#     zi = zi.reshape(xi.shape)

#     # Optional wedge-area normalization
#     if normalize_by_wedge:
#         # Avoid dividing by zero radius
#         R = np.maximum(xi, epsilon)
#         zi = zi / R

#     # Optionally save KDE grid and density to CSV
#     if kde_csv_path:
#         grid_data = pd.DataFrame({
#             'direct_distance': xi.flatten(),
#             'relative_angle':   yi.flatten(),
#             'density':          zi.flatten()
#         })
#         grid_data.to_csv(kde_csv_path, index=False)

#     # Plot
#     plt.figure(figsize=(10, 8))
#     contour = plt.contourf(xi, yi, zi, levels=num_levels, cmap='viridis')
#     plt.xlabel('Direct Distance (nm)')
#     plt.ylabel('Relative Angle (°)')
#     title = '2D KDE Contour Plot'
#     if normalize_by_wedge:
#         title += ' (Wedge‐Normalized)'
#     else:
#         title += ' (Raw)'
#     plt.title(title)
#     plt.colorbar(contour, label='Density')
#     plt.savefig(save_path)
#     plt.close()


# def plot_kde_contour_direct_angle(csv_filepath, save_path, kde_csv_path=None, num_levels=10):
#     """
#     Generate a 2D KDE contour plot for 'Direct Distances' and 'Relative Angle' using Gaussian KDE,
#     and optionally store the KDE evaluated grid values as a CSV file for later plotting.
    
#     Parameters:
#       csv_filepath: Path to acrossDSpacingInfo.csv.
#       save_path: Path to save the KDE contour plot image.
#       kde_csv_path: Optional; path to save the KDE grid and density values as a CSV file.
#       num_levels: Number of contour levels (default is 10).
#     """
#     df = pd.read_csv(csv_filepath)
#     direct = df['Direct Distances'].dropna().values
#     angle = df['Relative Angle'].dropna().values

#     # Define grid for KDE evaluation
#     xmin, xmax = direct.min(), direct.max()
#     ymin, ymax = angle.min(), angle.max()
#     xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
#     # Evaluate 2D Gaussian KDE on grid
#     values = np.vstack([direct, angle])
#     kernel = gaussian_kde(values)
#     zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
#     zi = zi.reshape(xi.shape)
    
#     # Optionally save KDE grid and density to CSV
#     if kde_csv_path is not None:
#         # Flatten the grid coordinates and density
#         grid_data = pd.DataFrame({
#             'x': xi.flatten(),
#             'y': yi.flatten(),
#             'density': zi.flatten()
#         })
#         grid_data.to_csv(kde_csv_path, index=False)
    
#     # Create the 2D contour plot
#     plt.figure(figsize=(10, 8))
#     contour = plt.contourf(xi, yi, zi, levels=num_levels, cmap='viridis')
#     plt.xlabel('Direct Distances (nm)')
#     plt.ylabel('Relative Angle (degrees)')
#     plt.title('2D KDE Contour Plot: Direct Distance vs. Relative Angle')
#     plt.colorbar(contour, label='Density')
#     plt.savefig(save_path)
#     plt.close()

def plot_edge_normalized_contour(csv_filepath, save_path,
                                 num_levels=10, r_min=1e-3,
                                 xlim=None, ylim=None,
                                 vmin=None, vmax=None,
                                 kde_csv_path=None):
    """
    Reads a CSV with columns:
       - 'Direct Distances'    (nm)
       - 'Relative Angle'      (degrees)
       - 'Radius1'             (nm)
       - 'Radius2'             (nm)
    Computes edge‐to‐edge distances, bins them with Freedman–Diaconis,
    builds a 2D histogram, normalizes by wedge area (r * dr * dtheta),
    optionally exports the grid and density to CSV, and saves a contour plot.

    Parameters:
      csv_filepath: path to the CSV containing the columns above.
      save_path:    path where the contour image will be written.
      num_levels:   number of contour levels in the plot.
      r_min:        minimum edge distance to avoid division by zero.
      xlim:         tuple (xmin, xmax) to set plt.xlim; None to auto.
      ylim:         tuple (ymin, ymax) to set plt.ylim; None to auto.
      vmin:         color scale minimum value; None to auto.
      vmax:         color scale maximum value; None to auto.
      kde_csv_path: if given, path to save the flattened grid & density as CSV.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Load and compute edge‐to‐edge distance
    df = pd.read_csv(csv_filepath)
    r_centroid = df['Direct Distances'].values
    r1 = df['Radius1'].values
    r2 = df['Radius2'].values
    r_edge = r_centroid - (r1 + r2)
    r_edge = np.clip(r_edge, r_min, None)
    theta = df['Relative Angle'].values

    # 2) Determine bins
    nbins_r     = freedman_diaconis_bins(pd.Series(r_edge))
    nbins_theta = freedman_diaconis_bins(pd.Series(theta))

    # 3) 2D histogram
    H, r_edges, th_edges = np.histogram2d(r_edge, theta,
                                         bins=[nbins_r, nbins_theta])

    # 4) Wedge‐area normalization
    r_centers  = (r_edges[:-1] + r_edges[1:]) / 2
    th_centers = (th_edges[:-1] + th_edges[1:]) / 2
    dr         = np.diff(r_edges)
    dth        = np.diff(th_edges)

    R, TH = np.meshgrid(r_centers, th_centers, indexing='ij')
    area = R * dr[:, None] * dth[None, :]
    H_norm = H / area

    # 4b) Optionally export flattened grid + density
    if kde_csv_path:
        flat = pd.DataFrame({
            'r_edge':  R.flatten(),
            'theta':   TH.flatten(),
            'density': H_norm.flatten()
        })
        flat.to_csv(kde_csv_path, index=False)

    # 5) Plot
    plt.figure(figsize=(10, 8))
    cf = plt.contourf(R, TH, H_norm,
                      levels=num_levels,
                      cmap='viridis')
    if vmin is not None or vmax is not None:
        cf.set_clim(vmin, vmax)

    plt.xlabel('Edge-to-Edge Distance (nm)')
    plt.ylabel('Relative Angle (°)')
    plt.title('Normalized 2D Density (edge distance vs angle)')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.colorbar(cf, label='Density / (nm·°)')
    plt.savefig(save_path)
    plt.close()




#############################
#          MAIN             #
#############################

if __name__ == "__main__":
    # Create and save the across-d-spacing info CSV
    across_df = createAcrossDSpacingInfo_csv()
    across_csv_path = join(DATA_DIR, "acrossDSpacingInfo.csv")
    across_df.to_csv(across_csv_path, index=False)
    print("Saved CSV:", across_csv_path)
    
    # Filter images and crystal pairs for peak and away regions
    peak_df, away_df = filter_images_and_crystals(across_csv_path,
                                                   peak_direct_range=(10,15), 
                                                   peak_angle_range=(5,10),
                                                   away_direct_thresh=35, 
                                                   away_angle_thresh=50)
    
    # Save the filtered results
    peak_csv = join(DATA_DIR, "peak_region.csv")
    away_csv = join(DATA_DIR, "away_region.csv")
    peak_df.to_csv(peak_csv, index=False)
    away_df.to_csv(away_csv, index=False)
    print("Saved peak region CSV:", peak_csv)
    print("Saved away region CSV:", away_csv)
    
    # Create and save the average angle info CSV
    avg_angle_df = createAvgAngle_acrossDSpacingInfo_csv()
    avg_angle_csv_path = join(DATA_DIR, "acrossDSpacingInfo_avg_angle.csv")
    avg_angle_df.to_csv(avg_angle_csv_path, index=False)
    print("Saved CSV:", avg_angle_csv_path)
    
    # Plot and save histograms for across-d-spacing info
    metric_direct_plot_path = join(DATA_DIR, "acrossDSpacingInfo_metric_direct_hist.png")
    plot_histogram_metric_direct(across_df, metric_direct_plot_path)
    print("Saved plot:", metric_direct_plot_path)
    
    relative_angle_plot_path = join(DATA_DIR, "acrossDSpacingInfo_relative_angle_hist.png")
    plot_histogram_relative_angle(across_df, relative_angle_plot_path)
    print("Saved plot:", relative_angle_plot_path)
    
    # Plot and save histograms for average angle info
    avg_angle_plot_path = join(DATA_DIR, "acrossDSpacingInfo_avg_angle_hist.png")
    plot_histogram_avg_relative_angle(avg_angle_df, avg_angle_plot_path)
    print("Saved plot:", avg_angle_plot_path)
    
    std_angle_plot_path = join(DATA_DIR, "acrossDSpacingInfo_std_angle_hist.png")
    plot_histogram_std_angles(avg_angle_df, std_angle_plot_path)
    print("Saved plot:", std_angle_plot_path)
    
    # Example usage of the histogram peak check functions:
    # Uncomment and update the CSV paths as needed:
    significant, p_val, info = check_histogram_peak_FD(avg_angle_csv_path, column='Avg Relative Angle')
    print(f"Peak significance: {significant} (p-value: {p_val})")
    print("Peak details:", info)
    
    # Uncomment to check histogram peak for a specific column in the across_df
    significant, p_val, info = check_histogram_peak_FD(across_csv_path, column='Relative Angle')
    print(f"Peak significance: {significant} (p-value: {p_val})")
    print("Peak details:", info)
    
    # Generate and save KDE plots
    kde_avg_angle_path = join(DATA_DIR, "kde_acrossDSpacingInfo_avg_angle.png")
    plot_kde_avg_relative_angle(avg_angle_df, kde_avg_angle_path)
    print("Saved KDE plot:", kde_avg_angle_path)
    
    kde_std_angle_path = join(DATA_DIR, "kde_acrossDSpacingInfo_std_angles.png")
    plot_kde_std_angles(avg_angle_df, kde_std_angle_path)
    print("Saved KDE plot:", kde_std_angle_path)
    
    kde_metric_direct_path = join(DATA_DIR, "kde_acrossDSpacingInfo_metric_direct.png")
    plot_kde_metric_direct(across_df, kde_metric_direct_path)
    print("Saved KDE plot:", kde_metric_direct_path)
    
    kde_relative_angle_path = join(DATA_DIR, "kde_acrossDSpacingInfo_relative_angle.png")
    plot_kde_relative_angle(across_df, kde_relative_angle_path)
    print("Saved KDE plot:", kde_relative_angle_path)
    
    # Generate and save 2D contour plots
    contour_plot_path = join(DATA_DIR, "contour_direct_vs_angle.png")
    plot_contour_direct_angle(across_csv_path, contour_plot_path, num_levels=4)
    print("Saved 2D contour plot:", contour_plot_path)
    
    # kde_csv_path = join(DATA_DIR, "kde_grid_data.csv")
    # kde_contour_plot_path_new = join(DATA_DIR, "kde_contour_direct_vs_angle.png")
    # plot_kde_contour_direct_angle(across_csv_path, kde_contour_plot_path_new, kde_csv_path=kde_csv_path, num_levels=10)
    # print("Saved 2D KDE contour plot with grid data:", kde_contour_plot_path_new)
    
    kde_zoomed_csv_path = join(DATA_DIR, "kde_zoomed_grid_data.csv")
    kde_zoomed_contour_plot_path_new = join(DATA_DIR, "kde_zoomed_contour_direct_vs_angle.png")
    plot_kde_contour_direct_angle(
        csv_filepath=across_csv_path,
        save_path=kde_zoomed_contour_plot_path_new,
        kde_csv_path=kde_zoomed_csv_path,
        num_levels=20,
        normalize_by_wedge=True,
        # xlim=None,
        # ylim=None,
        # vmin=None, 
        # vmax=None,
        xlim=(0.4, 1),    # show 0–30 nm only
        ylim=(60, 90),    # show 0–20 degrees only,
        vmin=1.25e-5, 
        vmax=6.0e-5,
        log_scale=False
      )
    
    across_df = createAcrossDSpacingInfo_with_radii_csv()
    across_csv_path = join(DATA_DIR, "acrossDSpacingInfo_with_radii.csv")
    across_df.to_csv(across_csv_path, index=False)
    
    # plot_edge_normalized_contour(
    # csv_filepath=across_csv_path,
    # save_path=join(DATA_DIR, "edge_normalized_contour.png"),
    # num_levels=10,
    # r_min=0.1
    # )
    
    plot_edge_normalized_contour(
    csv_filepath=across_csv_path,
    save_path=join(DATA_DIR, "edge_normalized_contour.png"),
    num_levels=8,
    r_min=0.1,
    # xlim=(0.4, 1),    # show 0–30 nm only
    # ylim=(60, 90),    # show 0–20 degrees only,
    # vmin=1.25e-5, 
    # vmax=6.0e-5,
    xlim=None,
    ylim=None,
    vmin=None,
    vmax=None,
    kde_csv_path=join(DATA_DIR, "edge_normalized_density.csv")
    )
    
    
