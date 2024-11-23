import os
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tabulate import tabulate

import xml_helpers as xh

# Check if the CSV file exists
# Get the workspace path
workspace_path = os.path.dirname(os.path.abspath(__file__))

# Construct the paths to the CSV file and XML directory
csv_file = os.path.join(workspace_path, "..", "data", "openlogo_data.csv")
xml_dir = os.path.join(workspace_path, "..", "data", "openlogo", "Annotations")

print(csv_file)
if os.path.exists(csv_file):
    # Load the parsed data from the CSV file
    data = pd.read_csv(csv_file)
else:
    # Parse all files
    data = xh.parse_xml_directory(xml_dir)
    # Save the parsed data to a CSV file
    data.to_csv(csv_file, index=False)

# Find the 5 most common names in the parsed XML data
top_5_names = data['name'].value_counts().head(5)
print("Top 5 names with the highest count:")
print(top_5_names)

print(data.head())

# Load a text file with file names to a separate dataset
text_file_path = os.path.join(workspace_path, "..", "data", "trainval.txt")

if os.path.exists(text_file_path):
    with open(text_file_path, 'r') as file:
        file_names = [line.strip() for line in file.readlines()]
    file_names_df = pd.DataFrame(file_names, columns=['file_name'])
    print("Loaded file names from text file:")
    print(file_names_df.head())
else:
    print(f"Text file {text_file_path} does not exist.")

# Count all names beginning with "cocacola"
cocacola_count = data[data['name'].str.startswith('cocacola')].shape[0]
print(f"Number of names beginning with 'cocacola': {cocacola_count}")

# Filter the data to include only rows with filenames contained in file_names_df, ignoring extensions
file_names_set = set(file_names_df['file_name'])
filtered_data = data[data['filename'].apply(lambda x: os.path.splitext(x)[0]).isin(file_names_set)]

print("Filtered data:")
print(filtered_data.head())

# Compare data to filtered_data
comparison = data.merge(filtered_data, on=['filename', 'name'], how='outer', indicator=True)
print("Comparison of data and filtered_data:")
print(comparison['_merge'].value_counts())

# Show examples from data that are not in filtered_data
not_in_filtered_data = data[~data['filename'].apply(lambda x: os.path.splitext(x)[0]).isin(file_names_set)]

print("Examples from data not in filtered_data:")
print(not_in_filtered_data.head())

# Count all names beginning with "cocacola"
cocacola_count = data[data['name'].str.startswith('cocacola')].shape[0]
print(f"Number of names beginning with 'cocacola': {cocacola_count}")



# # Display the full DataFrame without truncation
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

# # Display the data in a grid format
# print(tabulate(data.head(10), headers='keys', tablefmt='grid'))
                
# # let's move the images to the corresponding subfolders
# image_dir = os.path.join("..", "data", "openlogo", "JPEGImages")

# # Move images and annotations for the top 5 labels
# for label in tqdm(top_5_names.index, desc="Moving images and annotations by labels"):
#     xh.move_images_by_labels(data, xml_dir, image_dir, {label: label}, subfolder='..\data\top5')