import numpy as np
import matplotlib.pyplot as plt
import struct 

# Open the binary file in read mode
def open_images_dataset(path):
    with open(path, 'rb') as f:
        # Read the file header containing metadata
        magic_num = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')
    
        # Read the image data as a single array of bytes
        image_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape the image data into a 3D array
        images = image_data.reshape((num_images, num_rows, num_cols))
        return images

def open_labels_dataset(path):
    # Open the binary file in read mode
    with open(path, 'rb') as f:

        # Read the magic number and the number of items
        magic, num = struct.unpack(">II", f.read(8))
    
        # Read the labels
        labels = np.array(struct.unpack("B" * num, f.read(num)))
    return labels
