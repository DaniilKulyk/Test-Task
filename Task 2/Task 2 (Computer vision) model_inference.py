import pandas as pd
import numpy as np
import ast
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# Function to convert descriptors from string format to numpy array
def convert_descriptors(descriptor_string):
    try:
        # Convert the string back into a list of numbers
        descriptors = np.array(ast.literal_eval(descriptor_string), dtype=np.uint8)  # Set type to uint8
        return descriptors
    except:
        return None  # Return None if no descriptors are available


# Function to compare images using keypoints and descriptors
def compare_images(df, threshold=0.8, max_matches=None):

    # List to store matching image pairs and similarity percentage
    matches = []

    # Progress bar
    total_comparisons = (len(df) * (len(df) - 1)) // 2  # C(n, 2) = n * (n - 1) / 2
    progress_bar = tqdm(total=total_comparisons, desc="Comparing images", unit="comparison")

    # Brute-Force matcher for feature matching
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    # Loop through the DataFrame to compare each image with every other image
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            # Check if the limit of matches has been reached
            if max_matches is not None and len(matches) >= max_matches:
                print(f"Limit of {max_matches} matches reached. Stopping process.")
                progress_bar.close()
                return pd.DataFrame(matches, columns=["image1", "image2", "similarity (%)"])

            # Get descriptors of the two images to compare
            descriptors1 = convert_descriptors(df.loc[i, "descriptors"])
            descriptors2 = convert_descriptors(df.loc[j, "descriptors"])

            # Skip if no descriptors are found for either image
            if descriptors1 is None or descriptors2 is None:
                continue

            # Perform feature matching
            matches_found = matcher.knnMatch(descriptors1, descriptors2, 2)

            # Calculate the percentage of good matches (ratio of matches below a threshold)
            good_matches = [m for m, n in matches_found if m.distance < threshold * n.distance] #akaze
            match_ratio = len(good_matches) / max(len(matches_found), 1)  # Avoid division by zero
            match_percentage = match_ratio * 100  # Convert ratio to percentage

            # Add the match pair to the list along with the percentage
            matches.append([df.loc[i, "image_path"], df.loc[j, "image_path"], round(match_percentage, 2)])

            # Update progress bar
            progress_bar.update(1)

    progress_bar.close()

    # Convert the list of matches to a DataFrame for further analysis or saving
    matches_df = pd.DataFrame(matches, columns=["image1", "image2", "similarity (%)"])

    return matches_df


def convert_to_keypoints(kp_string):
    kp_list = ast.literal_eval(kp_string)

    keypoints = []
    for (pt, size, angle, response, octave, class_id) in kp_list:
        # Create an object cv2.KeyPoint
        keypoint = cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle, octave=octave, class_id=class_id)
        keypoints.append(keypoint)
    return keypoints


def load_and_preprocess_image_color(image_path, max_dimension=5490):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    scale = max_dimension / max(height, width)
    resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return resized_img


# Function to visualize matches between two images using keypoints from the DataFrame
def visualize_matches(df_keypoints, df_matches, match_index=0):

    # Extract the paths of the matching images from the df_matches DataFrame
    image1_path = df_matches.loc[match_index, 'image1']
    image2_path = df_matches.loc[match_index, 'image2']

    # Retrieve the keypoints and descriptors for the two images from df_keypoints
    kp1 = convert_to_keypoints(df_keypoints[df_keypoints['image_path'] == image1_path]['keypoints'].values[0])
    kp2 = convert_to_keypoints(df_keypoints[df_keypoints['image_path'] == image2_path]['keypoints'].values[0])

    des1 = convert_descriptors(df_keypoints[df_keypoints['image_path'] == image1_path]['descriptors'].values[0])
    des2 = convert_descriptors(df_keypoints[df_keypoints['image_path'] == image2_path]['descriptors'].values[0])

    # Load original images (format .jp2)
    original_img1_path = os.path.join('Sentinel2 IMG', os.path.basename(image1_path).replace('.jpg', '.jp2'))
    original_img2_path = os.path.join('Sentinel2 IMG', os.path.basename(image2_path).replace('.jpg', '.jp2'))

    original_img1 = load_and_preprocess_image_color(original_img1_path)
    original_img2 = load_and_preprocess_image_color(original_img2_path)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on their distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Plot images and matches

    plt.figure(figsize=(20, 10))

    # Display matches between images
    match_img = cv2.drawMatches(original_img1, kp1, original_img2, kp2, matches[:20], None, flags=2, matchesThickness=3)

    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title('Keypoint Matches on Original Images')

    plt.show()

    print(df_matches.loc[match_index, 'similarity (%)'])


# Read keypoints CSV
df_k = pd.read_csv('keypoints_data_mask.csv')

# Compare images in the DataFrame with a limit of 10 matches
matches_df = compare_images(df_k, threshold=0.8)

# Save the matching pairs to a CSV for later visualization
matches_df.to_csv('image_matches_mask.csv', index=False)

# Load matching pairs CSV
df_match = pd.read_csv('image_matches_mask.csv')

visualize_matches(df_k, df_match, match_index=0)
