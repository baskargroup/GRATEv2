import os
import argparse
from PIL import Image, ImageEnhance

def enhance_image(input_path, output_path, brightness_factor=2.0, contrast_factor=1.2):
    """
    Opens an image, enhances its brightness and contrast, then saves it.
    """
    # Open the image and convert to RGB (ensuring consistency)
    img = Image.open(input_path).convert("RGB")
    
    # Enhance brightness
    brightness_enhancer = ImageEnhance.Brightness(img)
    img_brightened = brightness_enhancer.enhance(brightness_factor)
    
    # Enhance contrast
    contrast_enhancer = ImageEnhance.Contrast(img_brightened)
    img_final = contrast_enhancer.enhance(contrast_factor)
    
    # Save the enhanced image
    img_final.save(output_path)
    print(f"Brightened image saved to {output_path}")

def process_directory(input_dir, output_dir, brightness_factor=2.0, contrast_factor=1.2):
    """
    Loops through all image files in the input directory, applies enhancements, 
    and saves them to the output directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Supported image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    # Loop through each file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, file_name)
            output_file_name = file_name
            output_path = os.path.join(output_dir, output_file_name)
            
            try:
                enhance_image(input_path, output_path, brightness_factor, contrast_factor)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhance HRTEM images by increasing brightness and contrast."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing the original image files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory where the enhanced images will be saved"
    )
    parser.add_argument(
        "--brightness", type=float, default=2.0,
        help="Brightness enhancement factor (default: 2.0, >1.0 increases brightness)"
    )
    parser.add_argument(
        "--contrast", type=float, default=1.2,
        help="Contrast enhancement factor (default: 1.2, >1.0 increases contrast)"
    )
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.brightness, args.contrast)

if __name__ == "__main__":
    main()
