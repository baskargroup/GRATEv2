import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
import pathlib as pl

# -----------------------
# User parameters
# -----------------------
# input_csv = "your_data.csv"  # Replace with your CSV file path
projectDir_fpath = pl.Path(__file__).parent.resolve()
input_csv_rpath = 'Results/RyanPaper/overall.csv'
input_csv = projectDir_fpath / input_csv_rpath
# input_csv = '/media/dgamdha/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/Results/RyanPaper/overall.csv'
bins = 50  # Number of bins for histogram
# -----------------------

# Load data
data = pd.read_csv(input_csv)
d_spacing = data["D-Spacing(FFT, nm)"].values
crystal_area = data["Crystal Area (nm^2)"].values

# Create a weighted histogram of d_spacing with weights = crystalArea
hist_vals, bin_edges = np.histogram(d_spacing, bins=bins, weights=crystal_area)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot the weighted frequency (crystalArea) vs. d_spacing
plt.figure(figsize=(8, 5))
plt.bar(bin_centers, hist_vals, width=(bin_edges[1]-bin_edges[0]), alpha=0.7, edgecolor='k')
plt.xlabel("d_spacing")
plt.ylabel("Total Crystal Area in Bin")
plt.title("Frequency Plot Weighted by Crystal Area")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------
# Fit Gaussian Mixture Models
# ---------------------------------------

# Prepare data for fitting
X = d_spacing.reshape(-1, 1)  # sklearn expects a 2D array

# Fit a 1-component Gaussian (single normal distribution)
gmm1 = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
gmm1.fit(X)
log_likelihood_1 = gmm1.score(X) * len(X)  # score gives avg log-likelihood per sample
aic_1 = gmm1.aic(X)
bic_1 = gmm1.bic(X)

# Fit a 2-component Gaussian mixture
gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm2.fit(X)
log_likelihood_2 = gmm2.score(X) * len(X)
aic_2 = gmm2.aic(X)
bic_2 = gmm2.bic(X)

# ---------------------------------------
# Likelihood Ratio Test
# ---------------------------------------
# Number of parameters for GMM in 1D:
# For k components: parameters = k means + k variances + (k-1) mixing weights = 3k - 1
# k=1 => params = 3*1 - 1 = 2
# k=2 => params = 3*2 - 1 = 5
params_1 = 2
params_2 = 5

LRT_stat = 2 * (log_likelihood_2 - log_likelihood_1)
df = params_2 - params_1  # degrees of freedom for the test
p_value = 1 - chi2.cdf(LRT_stat, df)

# Print results
print("Single Gaussian Model:")
print(f"  Log-Likelihood: {log_likelihood_1:.4f}")
print(f"  AIC: {aic_1:.4f}")
print(f"  BIC: {bic_1:.4f}")

print("\nTwo-Component Gaussian Mixture Model:")
print(f"  Log-Likelihood: {log_likelihood_2:.4f}")
print(f"  AIC: {aic_2:.4f}")
print(f"  BIC: {bic_2:.4f}")

print("\nLikelihood Ratio Test between 1-component and 2-component model:")
print(f"  LRT statistic: {LRT_stat:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  p-value: {p_value:.4f}")

# Interpretation:
# If p-value is small (e.g., < 0.05), the 2-component model is significantly better than the 1-component model.
