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
            file_data['filename_xml'] = filename
            data_list.append(file_data)
        except ET.ParseError as e:
            print(f"Error parsing {filename}: {e}")
            continue

    df = pd.DataFrame(data_list)
    return df

def move_images_by_labels(data, xml_path, image_path, label_dict, subfolder='top5'):
    """
    Move images to subfolders based on labels.
    """
    subfolder_path = os.path.join(image_path, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    
    for label, subfolder in label_dict.items():
        for index, row in tqdm(data.iterrows(), desc=f"Moving images for {label}", total=len(data)):
            if row['name'] == label:
                # Copy the image file
                src_path = os.path.join(image_path, row['filename'])
                dst_path = os.path.join(subfolder_path, row['filename'])
                shutil.copy(src_path, dst_path)

                # Copy the corresponding XML file
                xml_filename = row['filename_xml']
                xml_src_path = os.path.join(xml_path, xml_filename)
                xml_dst_path = os.path.join(subfolder_path, xml_filename)
                shutil.copy(xml_src_path, xml_dst_path)