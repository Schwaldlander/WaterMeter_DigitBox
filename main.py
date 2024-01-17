from google.colab import drive
drive.mount('/content/drive')
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import pandas as pd
import ast

# Define the path to the TSV file
annotations_path = '/content/drive/MyDrive/TlkWaterMeters/data.tsv'

# Read the annotations into a DataFrame
annotations_df = pd.read_csv(annotations_path, delimiter='\t', header=None)
annotations_df.columns = ['image_id', 'value', 'polygon']
annotations_df = annotations_df[1:]
# # Convert the polygon data from string to dictionary
annotations_df['polygon'] = annotations_df['polygon'].apply(ast.literal_eval)

# Here, you would continue to load images and parse polygon coordinates
# For the sake of example, we assume images are loaded into a list `images`
# and their corresponding polygons are parsed into `polygons`
# Parse annotations to extract polygon coordinates
def parse_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        data = file.readlines()

    totallen = len(annotations_df['polygon'])
    annotations = {}
    for i in range(1,1+totallen):
        # parts = line.strip().split('\t')
        image_id = annotations_df['image_id'][i]#parts[0]

        polygon = annotations_df['polygon'][i]['data']
        annotations[image_id] = polygon
    return annotations

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LSTM
from tensorflow.keras.models import Model

# Define constants
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Adjust as needed
NUM_VERTICES = 8  # Max number of vertices; adjust as needed
VERTEX_DIM = 2  # Each vertex has (x, y) coordinates

# Define a CNN + LSTM model for variable-length polygon vertices

