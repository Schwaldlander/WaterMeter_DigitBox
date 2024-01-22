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



import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
from PIL import Image
def visualize_model_predictions(data_loader, model, num_images=5):
    model.eval()  # Set the model to evaluation mode
    images_processed = 0

    with torch.no_grad():
        for images, true_masks in val_dataloader:
            if images_processed >= num_images:
                break

            # Get model predictions
            images = images.permute(0,3,1,2)
            predictions = model(images)

            for img, true_mask, pred in zip(images, true_masks, predictions):
                if images_processed >= num_images:
                    break

                img = img.cpu().numpy().transpose((1, 2, 0))
                print(img.shape)
                true_mask = true_mask.cpu().numpy().squeeze()
                pred_mask = pred.cpu().numpy().squeeze()

                # Plotting
                pred_mask[ :, 0] = (IMG_HEIGHT / 2) + (IMG_HEIGHT * pred_mask[ :, 0])  # x-coordinates
                pred_mask[:, 1] = (IMG_WIDTH / 2) + (IMG_WIDTH * pred_mask[ :, 1])  # y-coordinates
                print(true_mask)
                print(pred_mask)
                # plt.figure(figsize=(12, 4))
                fig, ax = plt.subplots()

                # plt.subplot(1, 3, 1)
                # plt.imshow(img)
                ax.imshow(img)
                polygon1 = patches.Polygon(pred_mask, linewidth=3, edgecolor='r', facecolor='none')
                polygon2 = patches.Polygon(true_mask, linewidth=3, edgecolor='g', facecolor='none')
                # Add the polygon to the Axes
                ax.add_patch(polygon1)
                ax.add_patch(polygon2)

                plt.title('Image Contrast')
                plt.show()
                # ax.show()

                images_processed += 1

# Example usage
visualize_model_predictions(data_loader, model, num_images=5)
