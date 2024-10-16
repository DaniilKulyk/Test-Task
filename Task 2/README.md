# Task 2. Computer vision. Sentinel-2 image matching

## Overview

The objective of the task is to compare satellite images, including moments such as the seasonal component.

## Data

The dataset provided on Kaggle ["Deforestation in Ukraine from Sentinel2 data"](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine/data?select=S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510) was chosen to work on the task.

## Solution

To solve this problem, the following steps were taken:
1. According to the recommendation to the task it was decided to use masks based on geojson data with selection of only certain key places on the images. 
2. Converting the images into grayscale and resizing images to save space and computing power.
3. Finding key points of images using AKAZE method and creating a dataframe where the information will be stored.
4. Comparison of two images based on the found dexrippers and key points. The results are also recorded in the dataframe.
5. Visualization of the obtained results for further analysis.

## Improvements

1. Use other models (SIFT, SURF, KAZE, FAST, BRIEF, ORB, ... etc.) to find key points, run tests and select the best one.
2. Use original image sizes for more information about the images.
3. Use Machine Learning to create an image recognition model.

## Working environment
The DataSpell IDE with Python 3.11 was used. All libraries used and their versions are listed in “requirements.txt”.
