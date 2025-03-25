import os
import cv2
import argparse

def convert_tif_to_png(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Loop through all files in the input directory
    for file_name in os.listdir(input_dir):
        # Process files ending with .tif or .tiff (case-insensitive)
        if file_name.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_dir, file_name)
            # Read the image using OpenCV
            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to read image: {input_path}")
                continue

            # Generate the output file path with .png extension
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(output_dir, base_name + ".png")
            
            # Save the image in PNG format
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"Converted {input_path} -> {output_path}")
            else:
                print(f"Failed to write image: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert .tif files to .png format using OpenCV"
    )
    parser.add_argument(
        "input_dir", type=str, help="Path to the directory containing .tif files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to the directory to save converted .png files"
    )
    args = parser.parse_args()
    
    convert_tif_to_png(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
