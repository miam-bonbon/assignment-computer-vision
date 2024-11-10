import os
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tabulate import tabulate

import xml_helpers as xh

# Check if the CSV file exists
csv_file = os.path.join("..", "data", "openlogo_data.csv")
xml_dir = os.path.join("..", "data", "openlogo", "Annotations")
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

# # Display the full DataFrame without truncation
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

# # Display the data in a grid format
# print(tabulate(data.head(10), headers='keys', tablefmt='grid'))
                
# let's move the images to the corresponding subfolders
image_dir = os.path.join("..", "data", "openlogo", "JPEGImages")

# Move images and annotations for the top 5 labels
for label in tqdm(top_5_names.index, desc="Moving images and annotations by labels"):
    xh.move_images_by_labels(data, xml_dir, image_dir, {label: label}, subfolder='..\data\top5')