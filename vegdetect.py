import rasterio
import numpy as np
import matplotlib.pyplot as plt

def display_original_image(image_path, save_path):
    """
    Display the original image and save it to a file.

    Args:
    - image_path (str): Path to the input image file.
    - save_path (str): Path to save the displayed image.

    Returns:
    - None
    """
    with rasterio.open(image_path) as src:
        plt.figure(figsize=(8, 8))
        plt.imshow(src.read([1, 2, 3]).transpose(1, 2, 0))
        plt.axis('off')
        plt.title("Original Image")
        plt.savefig(save_path)
        plt.show()

def display_color_channels(image_path, save_path):
    """
    Display and save 4 subplots on one plot, each visualizing one color channel.

    Args:
    - image_path (str): Path to the input image file.
    - save_path (str): Path to save the displayed image.

    Returns:
    - None
    """
    with rasterio.open(image_path) as src:
        plt.figure(figsize=(12, 8))
        for i in range(1, 5):
            plt.subplot(2, 2, i)
            plt.imshow(src.read(i), cmap='gray')
            plt.title(f'Channel {i}')
            plt.axis('off')
        plt.suptitle("Color Channels")
        plt.savefig(save_path)
        plt.show()

def display_false_color_image(image_path, save_path):
    """
    Display and save a false color image.

    Args:
    - image_path (str): Path to the input image file.
    - save_path (str): Path to save the displayed image.

    Returns:
    - None
    """
    with rasterio.open(image_path) as src:
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        false_color_image = np.stack((red, green, blue), axis=-1)
        plt.figure(figsize=(8, 8))
        plt.imshow(false_color_image)
        plt.axis('off')
        plt.title("False Color Image")
        plt.savefig(save_path)
        plt.show()

def compute_ndvi(image_path):
    """
    Compute the NDVI for each pixel in the image.

    Args:
    - image_path (str): Path to the input image file.

    Returns:
    - ndvi (numpy.ndarray): NDVI array.
    """
    with rasterio.open(image_path) as src:
        red = src.read(3).astype(float)
        nir = src.read(4).astype(float)
        ndvi = (nir - red) / (nir + red)
    return ndvi

def display_ndvi(ndvi, save_path, cmap='RdYlGn'):
    """
    Display and save a visualization of the NDVI.

    Args:
    - ndvi (numpy.ndarray): NDVI array.
    - save_path (str): Path to save the displayed image.
    - cmap (str): Name of the colormap (default: 'RdYlGn').

    Returns:
    - None
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(ndvi, cmap=cmap)
    plt.colorbar(label='NDVI')
    plt.axis('off')
    plt.title("NDVI Visualization")
    plt.savefig(save_path)
    plt.show()

def apply_ndvi_threshold(ndvi, threshold, save_path):
    """
    Generate and save a plot where pixels above the threshold are green.

    Args:
    - ndvi (numpy.ndarray): NDVI array.
    - threshold (float): NDVI threshold value between -1 and 1.
    - save_path (str): Path to save the displayed image.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.bwr  # Using a diverging colormap
    cmap.set_bad(color='gray')  # Set pixels outside threshold range to gray
    plt.imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.axis('off')
    plt.title("NDVI Threshold")
    plt.axhline(0.5, color='black', linestyle='--')  # Visualize the threshold line
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    image_path = "west_campus.tif"
    
    # Display original image
    display_original_image(image_path, "original_image.png")
    
    # Display color channels
    display_color_channels(image_path, "color_channels.png")
    
    # Display false color image
    display_false_color_image(image_path, "false_color_image.png")
    
    # Compute NDVI
    ndvi = compute_ndvi(image_path)
    
    # Display NDVI
    display_ndvi(ndvi, "ndvi_visualization.png")
    
    # Apply NDVI threshold
    apply_ndvi_threshold(ndvi, 0.5, "ndvi_threshold.png")