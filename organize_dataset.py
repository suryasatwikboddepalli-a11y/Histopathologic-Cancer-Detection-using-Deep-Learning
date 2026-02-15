# organize_dataset.py
import pandas as pd
import os
import shutil

# --- Configuration ---
# We've replaced the relative paths with the full, absolute paths you provided.
# NOTE: We use forward slashes '/' which work on all operating systems.
CSV_FILE_PATH = 'D:/Projects/Datasets/train_labels.csv'
SOURCE_IMAGE_FOLDER = 'D:/Projects/Datasets/train'
ORGANIZED_FOLDER = 'D:/Projects/Datasets/organized_train_data' # This new folder will be created here.

print("Starting dataset organization...")
print(f"Reading labels from: {CSV_FILE_PATH}")
print(f"Reading images from: {SOURCE_IMAGE_FOLDER}")

# --- Create destination directories ---
if not os.path.exists(ORGANIZED_FOLDER):
    os.makedirs(ORGANIZED_FOLDER)

path_cancer = os.path.join(ORGANIZED_FOLDER, 'cancer')
path_no_cancer = os.path.join(ORGANIZED_FOLDER, 'no_cancer')

if not os.path.exists(path_cancer):
    os.makedirs(path_cancer)
if not os.path.exists(path_no_cancer):
    os.makedirs(path_no_cancer)
    
# --- Read the CSV and move files ---
try:
    labels_df = pd.read_csv(CSV_FILE_PATH)
    moved_count = 0
    total_files = len(labels_df)

    print(f"Found {total_files} labels. Starting to move files...")

    for index, row in labels_df.iterrows():
        image_id = row['id']
        label = row['label']
        
        source_path = os.path.join(SOURCE_IMAGE_FOLDER, f"{image_id}.tif")
        
        if os.path.exists(source_path):
            if label == 1:
                destination_path = os.path.join(path_cancer, f"{image_id}.tif")
            else:
                destination_path = os.path.join(path_no_cancer, f"{image_id}.tif")
            
            shutil.move(source_path, destination_path)
            moved_count += 1

        if (index + 1) % 10000 == 0:
            print(f"Processed {index + 1}/{total_files} files...")

    print(f"\nOrganization complete! Moved {moved_count} files into '{ORGANIZED_FOLDER}'.")

except FileNotFoundError:
    print(f"Error: A file or folder was not found. Please double-check these paths:")
    print(f"1. CSV Path: {CSV_FILE_PATH}")
    print(f"2. Image Folder Path: {SOURCE_IMAGE_FOLDER}")