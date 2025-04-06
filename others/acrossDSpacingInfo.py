import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join, splitext
from utils import getAngleDifference, filterThreshArea
from scipy.stats import poisson

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


def createAcrossDSpacingInfo():
    
    projectPath     = os.path.dirname(os.path.abspath(__file__))
    dataDir         = 'Results/ryan/original_analysis/Results_v3'
    d_spaceDir1     = join(projectPath, dataDir, "1p9/CSV")       ## Set 1.9nm Directory Here
    d_spaceDir2     = join(projectPath, dataDir, "0p7/CSV")       ## Set 0.7nm Directory Here
    pix2nm          = 78.5

    files1 = [f for f in listdir(d_spaceDir1) if splitext(f)[1] == ".csv"]
    files2 = [f for f in listdir(d_spaceDir2) if splitext(f)[1] == ".csv"]
    commonCSV   = listIntersection(files1, files2)

    print("len files1:  ",len(files1))
    print("len files2:  ",len(files2))
    print("len commonCSV", len(commonCSV))

    MetricDistances = []
    DirectDistances = []
    ModRelAngle     = []

    for filename in commonCSV:
        if filename == "overall.csv":
            continue
        # print("File name:   ", filename)
        df1 = pd.read_csv(join(d_spaceDir1, filename))
        df1 = filterThreshArea(df1, 
                            'Crystal Area (nm^2)', 
                            d_space = 1.9,   # same unit as area column
                            threshold_area_factor=7)
        
        df2 = pd.read_csv(join(d_spaceDir2, filename))
        df2 = filterThreshArea(df2, 
                            'Crystal Area (nm^2)',
                            d_space = 0.7,   # same unit as area column
                            threshold_area_factor=10)

        for ind1, row1 in df1.iterrows():
            centroid1   = numericFromString(row1['Centroid'], pix2nm) 
            area1       = float(row1['Crystal Area (nm^2)'])
            rad1        = radFromArea(area1)
            ang1        = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])

            for ind2, row2 in df2.iterrows():
                centroid2   = numericFromString(row2['Centroid'], pix2nm) 
                area2       = float(row2['Crystal Area (nm^2)'])
                rad2        = radFromArea(area2)
                ang2        = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])

                CCdist      = centroidDist(centroid1, centroid2)
                MetricDist   = CCdist/ (rad1 + rad2)
                
                MetricDistances.append(MetricDist)                              ## Metric Distance
                DirectDistances.append(CCdist)                                  ## Direct Distance
                ModRelAngle.append(getAngleDifference(ang1, ang2))              ## Absolute Value of the Angle Difference

    print("len distances:   ", len(DirectDistances))

    df_dist = pd.DataFrame(list(zip(MetricDistances, DirectDistances, ModRelAngle)), columns=['Metric Distances','Direct Distances','Relative Angle'])
    df_dist = df_dist.round(2)
    df_dist.to_csv(join(projectPath, dataDir, "acrossDSpacingInfo.csv"))

def createAvgAngle_acrossDSpacingInfo():
    
    projectPath     = os.path.dirname(os.path.abspath(__file__))
    dataDir         = 'Results/ryan/original_analysis/Results_v3'
    d_spaceDir1     = join(projectPath, dataDir, "1p9/CSV")       ## Set 1.9nm Directory Here
    d_spaceDir2     = join(projectPath, dataDir, "0p7/CSV")       ## Set 0.7nm Directory Here
    pix2nm          = 78.5

    files1 = [f for f in listdir(d_spaceDir1) if splitext(f)[1] == ".csv"]
    files2 = [f for f in listdir(d_spaceDir2) if splitext(f)[1] == ".csv"]
    commonCSV   = listIntersection(files1, files2)

    print("len files1:  ",len(files1))
    print("len files2:  ",len(files2))
    print("len commonCSV", len(commonCSV))

    # MetricDistances = []
    # DirectDistances = []
    ModAvgRelAngle     = []
    ModStdRelAngle_1p9 = []
    ModStdRelAngle_0p7 = []

    for filename in commonCSV:
        if filename == "overall.csv":
            continue
        # print("File name:   ", filename)
        df1 = pd.read_csv(join(d_spaceDir1, filename))
        df1 = filterThreshArea(df1, 
                            'Crystal Area (nm^2)', 
                            d_space = 1.9,   # same unit as area column
                            threshold_area_factor=7)
        
        df2 = pd.read_csv(join(d_spaceDir2, filename))
        df2 = filterThreshArea(df2, 
                            'Crystal Area (nm^2)',
                            d_space = 0.7,   # same unit as area column
                            threshold_area_factor=10)
        
        df1_ang_avg = df1['Crystal Angle (zero at X-axis and clockwise positive)'].mean()
        df2_ang_avg = df2['Crystal Angle (zero at X-axis and clockwise positive)'].mean()
        df1_ang_std = df1['Crystal Angle (zero at X-axis and clockwise positive)'].std()
        df2_ang_std = df2['Crystal Angle (zero at X-axis and clockwise positive)'].std()
        
        ModAvgRelAngle.append(getAngleDifference(df1_ang_avg, df2_ang_avg))              ## Absolute Value of the Angle Difference
        ModStdRelAngle_1p9.append(df1_ang_std)
        ModStdRelAngle_0p7.append(df2_ang_std)
    
    print("len distances:   ", len(ModAvgRelAngle))
    df_dist = pd.DataFrame(list(zip(ModAvgRelAngle, ModStdRelAngle_1p9, ModStdRelAngle_0p7)), columns=['Avg Relative Angle','Std Angle 1.9nm','Std Angle 0.7nm'])
    df_dist = df_dist.round(2)
    df_dist.to_csv(join(projectPath, dataDir, "acrossDSpacingInfo_avg_angle.csv"))
    
    # Determine number of bins for Avg Relative Angle using Freedman-Diaconis rule
    bins_avg_rel = freedman_diaconis_bins(df_dist['Avg Relative Angle'])
    plt.figure(figsize=(10, 5))
    plt.hist(df_dist['Avg Relative Angle'], bins=bins_avg_rel, alpha=0.7, label='Avg Relative Angle')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Average Relative Angles')
    plt.legend()
    plt.savefig(join(projectPath, dataDir, "acrossDSpacingInfo_avg_angle_hist.png"))
    plt.close()
    
    # For the standard deviation histograms, compute a common binning from the combined data
    combined_std = pd.concat([df_dist['Std Angle 1.9nm'], df_dist['Std Angle 0.7nm']])
    bins_std = freedman_diaconis_bins(combined_std)
    plt.figure(figsize=(10, 5))
    plt.hist(df_dist['Std Angle 1.9nm'], bins=bins_std, alpha=0.7, label='Std Angle 1.9nm')
    plt.hist(df_dist['Std Angle 0.7nm'], bins=bins_std, alpha=0.7, label='Std Angle 0.7nm')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Standard Deviations of Angles')
    plt.legend()
    plt.savefig(join(projectPath, dataDir, "acrossDSpacingInfo_std_angle_hist.png"))
    plt.close()
    
    # # Plotting the histogram of average relative angles and standard deviations
    # plt.figure(figsize=(10, 5))
    # plt.hist(df_dist['Avg Relative Angle'], bins=30, alpha=0.7, label='Avg Relative Angle')
    # plt.xlabel('Angle (degrees)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Average Relative Angles')
    # plt.legend()
    # # plt.show()
    # # Save the plot
    # plt.savefig(join(projectPath, dataDir, "acrossDSpacingInfo_avg_angle_hist.png"))
    # plt.close()
    
    # # Plotting the histogram of average relative angles and standard deviations
    # plt.figure(figsize=(10, 5))
    # plt.hist(df_dist['Std Angle 1.9nm'], bins=30, alpha=0.7, label='Std Angle 1.9nm')
    # plt.hist(df_dist['Std Angle 0.7nm'], bins=30, alpha=0.7, label='Std Angle 0.7nm')
    # plt.xlabel('Angle (degrees)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Standard Deviations of Angles')
    # plt.legend()
    # # plt.show()
    # # Save the plot
    # plt.savefig(join(projectPath, dataDir, "acrossDSpacingInfo_std_angle_hist.png"))
    # plt.close()

# def plotHistogram():
#     projectPath     = os.path.dirname(os.path.abspath(__file__))
#     dataDir         = 'Results/ryan/original_analysis/Results_v3'
#     filename        = "acrossDSpacingInfo_M1.csv"
#     df_dist         = pd.read_csv(join(projectPath, dataDir, filename))
    
#     plt.figure(figsize=(10, 5))
#     plt.hist(df_dist['Metric Distances'], bins=30, alpha=0.7, label='Metric Distances')
#     plt.hist(df_dist['Direct Distances'], bins=30, alpha=0.7, label='Direct Distances')
#     plt.xlabel('Distance (nm)')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Metric and Direct Distances')
#     plt.legend()
#     # plt.show()
#     # Save the plot
#     plt.savefig(join(projectPath, dataDir, filename.replace('.csv', '_distance_hist.png')))
#     plt.close()
    
#     # plot histogram of relative angles
#     plt.figure(figsize=(10, 5))
#     plt.hist(df_dist['Relative Angle'], bins=30, alpha=0.7, label='Relative Angle')
#     plt.xlabel('Angle (degrees)')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Relative Angles')
#     plt.legend()
#     # plt.show()
#     # Save the plot
#     plt.savefig(join(projectPath, dataDir, filename.replace('.csv', '_angle_hist.png')))
#     plt.close()

def plotHistogram():
    projectPath     = os.path.dirname(os.path.abspath(__file__))
    dataDir         = 'Results/ryan/original_analysis/Results_v3'
    filename        = "acrossDSpacingInfo.csv"
    df_dist         = pd.read_csv(join(projectPath, dataDir, filename))
    
    # For Metric and Direct Distances, determine a common binning from the combined data.
    combined_dist = pd.concat([df_dist['Metric Distances'], df_dist['Direct Distances']])
    bins_dist = freedman_diaconis_bins(combined_dist)
    
    plt.figure(figsize=(10, 5))
    plt.hist(df_dist['Metric Distances'], bins=bins_dist, alpha=0.7, label='Metric Distances')
    plt.hist(df_dist['Direct Distances'], bins=bins_dist, alpha=0.7, label='Direct Distances')
    plt.xlabel('Distance (nm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Metric and Direct Distances')
    plt.legend()
    plt.savefig(join(projectPath, dataDir, filename.replace('.csv', '_distance_hist.png')))
    plt.close()
    
    # For Relative Angle, compute the binning using the Freedman-Diaconis rule.
    bins_rel = freedman_diaconis_bins(df_dist['Relative Angle'])
    plt.figure(figsize=(10, 5))
    plt.hist(df_dist['Relative Angle'], bins=bins_rel, alpha=0.7, label='Relative Angle')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Relative Angles')
    plt.legend()
    plt.savefig(join(projectPath, dataDir, filename.replace('.csv', '_angle_hist.png')))
    plt.close()
    
def freedman_diaconis_bins(data):
    """
    Calculate the optimal number of bins using the Freedman-Diaconis rule.
    Any NaN values in the data are removed before calculation.
    
    Parameters:
      data: 1D array-like numerical data.
      
    Returns:
      bins: The computed number of bins (an integer).
    """
    # Remove NaN values if data is a Pandas Series or a NumPy array
    if isinstance(data, pd.Series):
        data = data.dropna().values
    else:
        data = np.array(data)
        data = data[~np.isnan(data)]
    
    if len(data) < 2:
        return 1  # Not enough data to form a histogram.
    
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        # If IQR is zero, fall back to a simple rule.
        return int(np.ceil(np.sqrt(len(data))))
    
    bin_width = 2 * iqr / np.power(len(data), 1/3)
    data_range = data.max() - data.min()
    if bin_width == 0 or data_range == 0:
        return int(np.ceil(np.sqrt(len(data))))
    
    bins = int(np.ceil(data_range / bin_width))
    return bins
    

def check_histogram_peak(csv_filepath, column='value', num_bins=20, significance_level=0.05):
    """
    Reads a CSV file, computes a histogram for a given column,
    and determines if the highest peak is statistically significant.
    The significance is evaluated by comparing the observed peak count against
    the expected count (mean of bin counts) assuming a Poisson distribution.
    
    Parameters:
      csv_filepath: Path to the CSV file.
      column: The column name to analyze (default is 'value').
      num_bins: Number of bins for the histogram (default is 20).
      significance_level: p-value threshold for significance (default is 0.05).
    
    Returns:
      is_significant: Boolean indicating if the peak is statistically significant.
      p_value: The computed p-value.
      details: A dictionary with additional details (peak_count, expected_count, counts, bin_edges, peak_index).
    """
    df = pd.read_csv(csv_filepath)
    data = df[column].dropna()
    
    # Compute histogram
    counts, bin_edges = np.histogram(data, bins=num_bins)
    
    # Identify the peak bin
    peak_index = np.argmax(counts)
    peak_count = counts[peak_index]
    
    # Estimate expected count as the mean count per bin
    expected_count = np.mean(counts)
    
    # Compute p-value using the Poisson survival function (P(X >= peak_count))
    p_value = poisson.sf(peak_count - 1, expected_count)
    
    # Plot the histogram and highlight the peak bin
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvspan(bin_edges[peak_index], bin_edges[peak_index+1], color='red', alpha=0.3, label='Peak bin')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column} with Peak Highlighted')
    plt.legend()
    # plt.show()
    plt.savefig(csv_filepath.replace('.csv', '_histogram.png'))
    plt.close()
    
    is_significant = p_value < significance_level
    details = {
        'peak_count': peak_count,
        'expected_count': expected_count,
        'p_value': p_value,
        'counts': counts,
        'bin_edges': bin_edges,
        'peak_index': peak_index,
    }
    
    return is_significant, p_value, details

def check_histogram_peak_FD(csv_filepath, column='value', significance_level=0.05):
    """
    Reads a CSV file, computes a histogram for a given column using the Freedman-Diaconis binning strategy,
    and determines if the highest peak is statistically significant.
    The significance is evaluated by comparing the observed peak count against the expected count (mean of bin counts)
    assuming a Poisson distribution.
    
    Parameters:
      csv_filepath: Path to the CSV file.
      column: The column name to analyze (default is 'value').
      significance_level: p-value threshold for significance (default is 0.05).
    
    Returns:
      is_significant: Boolean indicating if the peak is statistically significant.
      p_value: The computed p-value.
      details: A dictionary with additional details (peak_count, expected_count, counts, bin_edges, peak_index).
    """
    df = pd.read_csv(csv_filepath)
    data = df[column].dropna()
    
    # Determine the number of bins using the Freedman-Diaconis rule
    num_bins = freedman_diaconis_bins(data)
    print(f"Using {num_bins} bins (Freedman-Diaconis rule) for the histogram.")
    
    # Compute histogram
    counts, bin_edges = np.histogram(data, bins=num_bins)
    
    # Identify the peak bin
    peak_index = np.argmax(counts)
    peak_count = counts[peak_index]
    
    # Estimate expected count as the mean count per bin
    expected_count = np.mean(counts)
    
    # Compute p-value using the Poisson survival function (P(X >= peak_count))
    p_value = poisson.sf(peak_count - 1, expected_count)
    
    # Plot the histogram and highlight the peak bin
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvspan(bin_edges[peak_index], bin_edges[peak_index+1], color='red', alpha=0.3, label='Peak bin')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column} with Peak Highlighted')
    plt.legend()
    plt.show()
    
    is_significant = p_value < significance_level
    details = {
        'peak_count': peak_count,
        'expected_count': expected_count,
        'p_value': p_value,
        'counts': counts,
        'bin_edges': bin_edges,
        'peak_index': peak_index,
    }
    
    return is_significant, p_value, details
    
    
if __name__ == "__main__":
    createAcrossDSpacingInfo()
    plotHistogram()
    createAvgAngle_acrossDSpacingInfo()
    
    # Example usage of check_histogram_peak:
    # projectPath     = os.path.dirname(os.path.abspath(__file__))
    # dataDir         = 'Results/ryan/original_analysis/Results_v3'
    
    # ---
    # csv_path     = join(projectPath, dataDir, "acrossDSpacingInfo_avg_angle.csv")
    # # significant, p_val, info = check_histogram_peak(csv_path, column='Avg Relative Angle')
    # significant, p_val, info = check_histogram_peak_FD(csv_path, column='Avg Relative Angle')
    # ---
    
    # ---
    # csv_path     = join(projectPath, dataDir, "acrossDSpacingInfo.csv")       
    # significant, p_val, info = check_histogram_peak(csv_path, column='Relative Angle')
    # significant, p_val, info = check_histogram_peak_FD(csv_path, column='Relative Angle')
    # ---
    
    # print(f"Is the histogram peak statistically significant? {significant} (p-value: {p_val})")
    
    # # Print other info
    # print("Peak Count:", info['peak_count'])
    # print("Expected Count:", info['expected_count'])
    # print("Counts:", info['counts'])
    # print("Bin Edges:", info['bin_edges'])
    # print("Peak Index:", info['peak_index'])

