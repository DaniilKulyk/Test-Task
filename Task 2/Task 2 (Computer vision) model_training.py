import pandas as pd
import numpy as np
import os
import shutil
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cv2
import glob
from tqdm import tqdm
from pyproj import CRS


def extract_tci_images(source_dir, target_dir, max_images=None):
    # Create a folder for the extracted images if it does not already exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Counter to limit the number of files
    count = 0

    # Go through all the files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_TCI.jp2"):  # Check that the file ends with “_TCI.jp2”.
                full_path = os.path.join(root, file)
                shutil.copy(full_path, target_dir)  # Copy the file to a new folder
                count += 1

                if max_images and count >= max_images:
                    print(f"Extracted {count} of images, stop.")
                    return

    print(f"A total of {count} images have been extracted.")


# Loading an image and resizing
def load_and_preprocess_image(image_path, max_dimension=5490):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    scale = max_dimension / max(height, width)
    resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return resized_img


# Convert polygons to image coordinates
def poly_from_utm(polygon, transform):
    poly = unary_union(polygon)
    poly_pts = [~transform * tuple(coord[:2]) for coord in np.array(poly.exterior.coords)]
    return Polygon(poly_pts)


# Create a mask for an image based on geodata
def create_mask_from_geodata(image_path, geodata_path):
    # Open the image with rasterio
    with rasterio.open(image_path) as src:
        transform = src.meta['transform']
        img_size = (src.meta['height'], src.meta['width'])

    # Reading geodata (.geojson file)
    df = gpd.read_file(geodata_path)

    # Set the coordinate system
    df.crs = CRS.from_epsg(4236)  # Use the CRS.from_epsg format
    df = df.to_crs(CRS.from_string(src.crs.to_string()))  # Transforming polygons to the raster crs

    # Convert polygons to image coordinates
    poly_shp = []
    for _, row in df.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            poly_shp.append(poly_from_utm(geom, transform))
        elif geom.geom_type == 'MultiPolygon':
            for p in geom.geoms:
                poly_shp.append(poly_from_utm(p, transform))

    # Creating a mask using the rasterize function
    mask = rasterize(shapes=poly_shp, out_shape=img_size)

    # Convert the mask to uint8 format
    mask = (mask * 255).astype(np.uint8)

    return mask


# Creating a mask for pictures
def preprocess_and_save_images_with_masks(source_images_path, dest_dataset_path, geodata_path, max_dimension=5490):
    if not os.path.exists(dest_dataset_path):
        os.makedirs(dest_dataset_path)

    # Getting all TCI images from a specified folder
    tci_image_files = glob.glob(os.path.join(source_images_path, "*_TCI.jp2"))

    for image_file in tci_image_files:
        # Getting the file name
        base_name = os.path.basename(image_file)

        # Constructing a path to save a preprocessed image
        dest_image_path = os.path.join(dest_dataset_path, base_name.replace('.jp2', '.jpg'))

        # Creating an image mask based on geodata
        mask = create_mask_from_geodata(image_file, geodata_path)

        # Image loading and preprocessing
        preprocessed_image = load_and_preprocess_image(image_file, max_dimension)

        # Resize the mask to fit the size of the preprocessed image
        mask_resized = cv2.resize(mask, (preprocessed_image.shape[1], preprocessed_image.shape[0]))

        # Applying a mask to a preprocessed image
        masked_image = cv2.bitwise_and(preprocessed_image, preprocessed_image, mask=mask_resized)

        # Saving masked image
        cv2.imwrite(dest_image_path, masked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# Function to process images and find keypoints using AKAZE
def process_images(images_folder, output_csv_path):

    # List to store image paths, keypoints, and descriptors
    data = []

    # Find all images with the "_TCI" suffix
    image_paths = glob.glob(os.path.join(images_folder, '*_TCI.*'))

    # Initialize AKAZE
    akaze = cv2.AKAZE_create()

    for image_path in image_paths:
        print(f'Processing {image_path}')

        # Open the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # grayscale

        # Detect keypoints and descriptors using AKAZE
        keypoints, descriptors = akaze.detectAndCompute(image, None)

        # Save keypoints and descriptors if they are found
        if keypoints is not None and descriptors is not None:
            # Convert keypoints to a simpler format for storage
            kp_list = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

            # Add the data to the list (image path, keypoints, descriptors)
            data.append({
                "image_path": image_path,
                "keypoints": kp_list,  # List of keypoint coordinates
                "descriptors": descriptors.tolist()  # Convert descriptors to list
            })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file for easy inspection
    df.to_csv(output_csv_path, index=False)
    print(f'Keypoints and descriptors saved to {output_csv_path}')


# Call functions to extract images, create masks and save keypoints
source_directory = 'Sentinel2/Sentinel2'
target_directory = 'Sentinel2 IMG'
extract_tci_images(source_directory, target_directory)

source_images_path = 'Sentinel2 IMG'
dest_dataset_path = 'Sentinel2 IMG Mask 5k'
geodata_path = 'Sentinel2/Sentinel2/deforestation_labels.geojson'
preprocess_and_save_images_with_masks(source_images_path, dest_dataset_path, geodata_path)

output_csv_path = 'keypoints_data_mask.csv'
process_images(dest_dataset_path, output_csv_path)
