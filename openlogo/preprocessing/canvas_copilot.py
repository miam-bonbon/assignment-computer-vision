import os
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tabulate import tabulate

def parse_flat_xml_structure(file_path):
    """Helper function to parse XML file and extract all leaf node values"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = {}
    
    def extract_leaf_nodes(element, data):
        for child in element:
            if len(child) == 0:  # Leaf node
                data[child.tag] = child.text
            else:
                extract_leaf_nodes(child, data)
    
    extract_leaf_nodes(root, data)
    return data

def parse_xml_directory(directory_path, file_count=None):
    """
    Parse XML files in a directory and return a DataFrame.

    Args:
        directory_path (str): Path to the directory containing XML files.
        file_count (int, optional): Number of files to process. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing parsed XML data.
    """
    data_list = []
    files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]
    if file_count is not None:
        files = files[:file_count]
    
    for filename in tqdm(files, desc="Parsing XML files"):
        file_path = os.path.join(directory_path, filename)
        try:
            file_data = parse_flat_xml_structure(file_path)
            file_data['filename'] = filename
            data_list.append(file_data)
        except ET.ParseError as e:
            print(f"Error parsing {filename}: {e}")
            continue

    df = pd.DataFrame(data_list)
    return df

# Parse all files
data = parse_xml_directory(r"C:\Users\CAS\Downloads\openlogo\Annotations")

# Find the 5 most common names in the parsed XML data
top_5_names = data['name'].value_counts().head(5)
print("Top 5 names with the highest count:")
print(top_5_names)

# Display the full DataFrame without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Display the data in a grid format
print(tabulate(data.head(10), headers='keys', tablefmt='grid'))

def move_images_by_labels(data, image_path, label_dict):
    """
    Move images to subfolders based on labels.
    """
    for label, subfolder in label_dict.items():
        subfolder_path = os.path.join(image_path, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        for index, row in data.iterrows():
            if row['label'] == label:
                src_path = os.path.join(image_path, row['filename'])
                dst_path = os.path.join(subfolder_path, row['filename'])
                shutil.move(src_path, dst_path)
                
# let's move the images to the corresponding subfolders
image_path = r"C:\Users\CAS\Downloads\openlogo\JPEGImages"

# Move images for the top 5 labels
for label in tqdm(top_5_names.index, desc="Moving images by labels"):
    move_images_by_labels(data, image_path, {label: label})