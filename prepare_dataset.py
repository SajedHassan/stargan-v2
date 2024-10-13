import h5py
import os
import numpy as np
from PIL import Image
import tifffile
import cv2 as cv

# Open the file
dataset_dir = '/Users/sajedalmorsy/Academic/Masters/thesis/Tumor dataset/MMIS2024TASK1/validation_2d_splitted/a0/'
target_dir = '/Users/sajedalmorsy/Academic/Masters/thesis/Tumor dataset/MMIS2024TASK1/validation_2d_splitted_img/a0/'

def normalize(image_array):
   # Normalize to [0, 1]
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    # Scale to [0, 255] and convert to uint8
    image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    return image_array

for sample_file in os.listdir(dataset_dir):
  if (sample_file.startswith('.')):
      continue
  with h5py.File(dataset_dir + sample_file, 'r') as f:
      # t1 = f['t1']
      t2 = np.array(f['t2'])
      t1c = np.array(f['t1c'])
      label = np.array(f['label'])

      t2_norm = normalize(t2)
      t1c_norm = normalize(t1c)
      label_norm = normalize(label)

      img_array = np.stack([t2_norm, t1c_norm, label_norm], axis=2)
      print(img_array.shape)
      image = Image.fromarray(img_array)
      image.save(target_dir + sample_file.split('.')[0] + '.png', format='PNG')
      os.remove(target_dir + sample_file.split('.')[0] + '.tif')
      # tifffile.imwrite(target_dir + sample_file.split('.')[0] + '.tif', img_array)
      print('----')
      # array = tifffile.imread(target_dir + sample_file.split('.')[0] + '.tif')
      # print('----')


  # # Path to the TIFF file
  # tiff_file_path = target_dir + sample_file.split('.')[0] + '.tif'
  # # Path to save the PNG file
  # png_file_path = target_dir + sample_file.split('.')[0] + '.png'

  # # # Open the TIFF file
  # # image_array = tifffile.imread(tiff_file_path)

  # # if image_array.dtype == 'float32':
  # #   # Normalize to [0, 1]
  # #   image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
  # #   # Scale to [0, 255] and convert to uint8
  # #   image_array = (image_array * 255).clip(0, 255).astype(np.uint8)

  # # image = Image.fromarray(image_array)
  # # image.save(png_file_path, format='PNG')
  # os.remove(png_file_path)
