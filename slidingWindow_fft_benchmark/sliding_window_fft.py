import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from skimage.filters import gaussian
from skimage.measure import find_contours
from scipy.signal.windows import hann

def create_synthetic_hrtem_image(size=(512, 512), d_spacing_nm=1.9, resolution_px_per_nm=78.5, angle_deg=30):
    """
    Creates a synthetic image with an oriented fringe pattern to simulate
    a crystalline region within an amorphous background.

    Args:
        size (tuple): The output image dimensions (height, width).
        d_spacing_nm (float): The distance between fringes in nanometers.
        resolution_px_per_nm (float): The image resolution in pixels per nanometer.
        angle_deg (float): The orientation of the fringes in degrees.

    Returns:
        numpy.ndarray: A grayscale floating-point image.
    """
    # Convert physical units to pixel units
    d_spacing_px = d_spacing_nm * resolution_px_per_nm
    frequency_px = 1 / d_spacing_px
    angle_rad = np.deg2rad(angle_deg)

    # Create coordinate grid
    x = np.linspace(-size[1] / 2, size[1] / 2, size[1])
    y = np.linspace(-size[0] / 2, size[0] / 2, size[0])
    xx, yy = np.meshgrid(x, y)

    # Rotate coordinates to align with fringe angle
    xx_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)

    # Create sinusoidal pattern (fringes)
    fringes = np.sin(2 * np.pi * frequency_px * xx_rot)

    # Create a mask for the crystalline region (e.g., a soft circle)
    mask = np.exp(-((xx**2 + yy**2) / (2 * (size[0] / 4)**2)))

    # Combine fringes and mask
    image = fringes * mask

    # Add realistic noise
    noise = np.random.normal(0, 0.3, size)
    image += noise
    
    # Normalize image to [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return util.img_as_ubyte(image)


def detect_crystals_fft(image, window_size=96, stride=16,
                        d_spacing_range_nm=(1.5, 2.5),
                        resolution_px_per_nm=78.5,
                        central_mask_radius=5,
                        threshold_quantile=0.90):
    """
    Detects crystalline regions in an HRTEM image using a sliding-window FFT approach.

    Args:
        image (numpy.ndarray): Input grayscale HRTEM image.
        window_size (int): The size of the sliding window in pixels.
        stride (int): The step size for the sliding window.
        d_spacing_range_nm (tuple): The expected range of d-spacing (min, max) in nanometers.
        resolution_px_per_nm (float): Image resolution in pixels per nm.
        central_mask_radius (int): Radius (in pixels) of the central mask to remove the DC component.
        threshold_quantile (float): Quantile for thresholding the final crystallinity map.

    Returns:
        tuple: A tuple containing:
            - list: The detected contour boundaries.
            - numpy.ndarray: The generated crystallinity map.
    """
    # 1. Prepare image
    if image.ndim > 2:
        image = color.rgb2gray(image)
    image = util.img_as_float(image)
    
    # 2. Calculate frequency range in pixel units for the FFT
    # The frequency k in the FFT output is related to window_size and d-spacing in pixels
    # k = window_size / d_spacing_px
    d_spacing_min_px = d_spacing_range_nm[0] * resolution_px_per_nm
    d_spacing_max_px = d_spacing_range_nm[1] * resolution_px_per_nm
    
    # The max frequency corresponds to the min d-spacing, and vice-versa
    k_max = window_size / d_spacing_min_px
    k_min = window_size / d_spacing_max_px

    print(f"Searching for FFT peaks between radii {k_min:.2f} and {k_max:.2f} pixels from center.")

    # 3. Initialize maps
    crystallinity_map = np.zeros_like(image, dtype=np.float64)
    normalization_map = np.zeros_like(image, dtype=np.float64)

    # 4. Create a Hanning window to apply to patches to reduce spectral leakage
    hanning_1d = hann(window_size)
    hanning_window = np.outer(hanning_1d, hanning_1d)

    # 5. Create frequency grid for the mask (do this once)
    u = np.fft.fftfreq(window_size) * window_size
    v = np.fft.fftfreq(window_size) * window_size
    uu, vv = np.meshgrid(u, v, indexing='ij')
    radius_grid = np.sqrt(uu**2 + vv**2)
    
    # Create the annular (ring) mask
    annulus_mask = (radius_grid >= k_min) & (radius_grid <= k_max)
    # Also mask the central DC component area
    annulus_mask[radius_grid < central_mask_radius] = False


    # 6. Slide window over the image
    for y in range(0, image.shape[0] - window_size, stride):
        for x in range(0, image.shape[1] - window_size, stride):
            # Extract window
            window = image[y:y + window_size, x:x + window_size]
            
            # Apply Hanning window function
            window = window * hanning_window

            # Compute 2D FFT and power spectrum
            fft_result = np.fft.fftshift(np.fft.fft2(window))
            power_spectrum = np.abs(fft_result)**2

            # Score the window by finding the max power within the annular mask
            score = np.max(power_spectrum[annulus_mask]) if np.any(power_spectrum[annulus_mask]) else 0
            
            # Add score to the map
            crystallinity_map[y:y + window_size, x:x + window_size] += score
            normalization_map[y:y + window_size, x:x + window_size] += 1
    
    # 7. Normalize the map where windows overlapped
    # Avoid division by zero in areas with no windows
    normalization_map[normalization_map == 0] = 1
    crystallinity_map /= normalization_map
    
    # 8. Post-processing
    # Smooth the map to connect regions
    smoothed_map = gaussian(crystallinity_map, sigma=window_size/4)
    
    # Add a check to prevent crashing if no features are found
    positive_values = smoothed_map[smoothed_map > 0]
    if positive_values.size == 0:
        print("Warning: No crystalline features were detected. Returning empty results.")
        return [], smoothed_map # Return empty contours and the zero-map

    # Threshold the map to get a binary mask of crystals
    threshold_val = np.quantile(positive_values, threshold_quantile)
    binary_mask = smoothed_map > threshold_val

    # Find contours of the detected regions
    contours = find_contours(binary_mask, level=0.5)

    return contours, crystallinity_map


if __name__ == '__main__':
    # --- Parameters for Detection ---
    D_SPACING_NM = 2.1
    RESOLUTION_PX_PER_NM = 78.5
    
    # --- NEW: Define the desired output image size in pixels ---
    OUTPUT_IMAGE_SIZE_PX = 600
    
    # --- Load Image ---
    try:
        image_to_process = io.imread('FoilHole_21830219_Data_21829764_21829765_20200122_1016.tif')
    except FileNotFoundError:
        print("Error: Your image file was not found. Please check the path.")
        exit()

    # --- Run Detection ---
    print("Starting crystal detection with sliding-window FFT...")
    detected_contours, crystallinity_map = detect_crystals_fft(
        image=image_to_process,
        window_size=640,
        stride=16,
        d_spacing_range_nm=(D_SPACING_NM - 1.4, D_SPACING_NM + 1.6),
        resolution_px_per_nm=RESOLUTION_PX_PER_NM,
        threshold_quantile=0.90
    )
    print(f"Detection complete. Found {len(detected_contours)} potential regions.")

    # --- Save Outputs ---
    if detected_contours: # Only save if something was detected
        # Import the resize function
        from skimage.transform import resize
        
        # 1. Resize and save the crystallinity map
        print(f"Saving crystallinity_map.jpg ({OUTPUT_IMAGE_SIZE_PX}x{OUTPUT_IMAGE_SIZE_PX})...")
        resized_map = resize(crystallinity_map, 
                             (OUTPUT_IMAGE_SIZE_PX, OUTPUT_IMAGE_SIZE_PX), 
                             anti_aliasing=True)
        # plt.imsave('crystallinity_map.jpg', resized_map, cmap='viridis')
        plt.imsave('swfft_cmap.jpg', resized_map, cmap='viridis')

        # 2. Save the image with detected crystal contours at the specified size
        print(f"Saving detected_crystals.jpg ({OUTPUT_IMAGE_SIZE_PX}x{OUTPUT_IMAGE_SIZE_PX})...")
        # Define DPI and calculate the figure size in inches to get the correct pixel output
        output_dpi = 100
        figsize_inches = OUTPUT_IMAGE_SIZE_PX / output_dpi
        
        # Create a new figure just for saving
        save_fig, save_ax = plt.subplots(figsize=(figsize_inches, figsize_inches))
        save_ax.imshow(image_to_process, cmap='gray')
        for contour in detected_contours:
            save_ax.plot(contour[:, 1], contour[:, 0], linewidth=2, c='r')
        save_ax.axis('off')
        
        # Save the figure with tight padding and the specified DPI
        save_fig.savefig('swfft_det.jpg', bbox_inches='tight', pad_inches=0, dpi=output_dpi)
        plt.close(save_fig) # Close the figure to prevent it from displaying with plt.show()

    # --- Visualization (for screen display) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes.ravel()

    ax[0].imshow(image_to_process, cmap='gray')
    ax[0].set_title('Input HRTEM Image')

    ax[1].imshow(crystallinity_map, cmap='viridis')
    ax[1].set_title('Crystallinity Map')

    ax[2].imshow(image_to_process, cmap='gray')
    ax[2].set_title(f'Detected Crystals ({len(detected_contours)} regions)')
    for contour in detected_contours:
        ax[2].plot(contour[:, 1], contour[:, 0], linewidth=2, c='r')
    
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()