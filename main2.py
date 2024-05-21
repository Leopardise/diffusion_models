import cv2
import os

# Directory containing the images
image_dir = 'output'

# Target size for resizing
target_size = (512, 512)

# Loop through the directories and resize images
for sequence_id in os.listdir(image_dir):
    sequence_path = os.path.join(image_dir, sequence_id)
    for image_file in os.listdir(sequence_path):
        image_path = os.path.join(sequence_path, image_file)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, target_size)
        cv2.imwrite(image_path, resized_image)
