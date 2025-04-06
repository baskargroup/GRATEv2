import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, splitext
from utils import getAngleDifference, filterThreshArea
from scipy.stats import poisson

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

#############################
#   CSV CREATION FUNCTIONS  #
#############################

def createAcrossDSpacingInfo_csv():
    """
    Process CSV files from both directories and create a DataFrame containing:
      - Metric Distances
      - Direct Distances
      - Relative Angle (absolute difference between crystal angles)
    
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
    
    print("Total pairs processed:", len(DirectDistances))
    df = pd.DataFrame(list(zip(MetricDistances, DirectDistances, ModRelAngle)),
                      columns=['Metric Distances', 'Direct Distances', 'Relative Angle'])
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

#############################
#          MAIN             #
#############################

if __name__ == "__main__":
    # Create and save the across-d-spacing info CSV
    across_df = createAcrossDSpacingInfo_csv()
    across_csv_path = join(DATA_DIR, "acrossDSpacingInfo.csv")
    across_df.to_csv(across_csv_path, index=False)
    print("Saved CSV:", across_csv_path)
    
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