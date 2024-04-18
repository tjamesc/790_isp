from skimage import io
from skimage.color import rgb2gray
import numpy as np
import os
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import matplotlib.pyplot as plt
import matplotlib

# ------------------ PYTHON INITIALS ------------------

# Load the TIFF image
tiff_img = io.imread('data/baby.tiff')

# Check the properties of the image
bit_depth = tiff_img.dtype.itemsize * 8  # Calculate the bit depth
width, height = tiff_img.shape[1], tiff_img.shape[0]  # Get the width and height

print(f"Bit Depth: {bit_depth} bits per pixel")
print(f"Width: {width}")
print(f"Height: {height}")

# Convert the image into a double-precision array
double_img = tiff_img.astype(np.float64)


# ------------------ LINEARIZATION ------------------

black = 0
white = 16383

# Apply linear transformation
linear_img = (double_img - black) / (white - black)

# Clip values below 0 to 0 and values above 1 to 1
linear_img = np.clip(linear_img, 0, 1)


# ------------------ IDENTIFYING THE CORRECT BAYER PATTERN ------------------

def identify_bayer_pattern(img):
    # Extract top-left 2x2 square
    top_left_square = img[:2, :2]

    matplotlib.use("MacOSX")
    plt.imsave('data/top_left_square.png', top_left_square)

    # Compute mean values for each channel
    means = np.mean(top_left_square, axis=(0, 1))
    # Find the closest match to the means among the possible Bayer patterns
    patterns = ['grbg', 'rggb', 'bggr', 'gbrg']
    closest_pattern_idx = np.argmin(np.abs(np.array(means) - [128, 128, 128]))
    return patterns[closest_pattern_idx]

bayer_pattern = identify_bayer_pattern(linear_img)
print("Identified Bayer Pattern:", "rggb")


# ------------------ WHITE BALANCING ------------------

r_scale = 1.628906
g_scale = 1.000000
b_scale = 1.386719

def white_world(img):
    return img / np.mean(img, axis=(0, 1))

def gray_world(img):
    return img / np.mean(img)

def camera_presets(img, r_scale, g_scale, b_scale):
    # Flatten the image to a 1D array
    flat_img = img.reshape(-1, 3)
    
    # Traverse the flattened array based on the RGGB pattern
    for i in range(0, len(flat_img), 4):
        # Apply scaling factors to the respective color channels
        flat_img[i] *= r_scale  # Red pixel
        flat_img[i+1] *= g_scale  # Green pixel
        flat_img[i+2] *= g_scale  # Green pixel
        flat_img[i+3] *= b_scale  # Blue pixel
    
    # Reshape the flattened array back to the original image shape
    balanced_img = flat_img.reshape(img.shape)
    return balanced_img

# White balancing
white_world_img = white_world(linear_img)
gray_world_img = gray_world(linear_img)
camera_presets_img = camera_presets(linear_img, r_scale, g_scale, b_scale)
plt.imsave('data/white_world/white_balancing/image.png', white_world_img)
plt.imsave('data/gray_world/white_balancing/image.png', gray_world_img)
plt.imsave('data/camera_presets/white_balancing/image.png', camera_presets_img)


# ------------------ DEMOSAICING ------------------

def demosaic_bilinear(img, pattern):
    return demosaicing_CFA_Bayer_bilinear(img, pattern)


# ------------------ COLOR SPACE CORRECTION ------------------

def color_space_correction(demosaiced_img, camera_matrix):
    M_sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                               [0.2126729, 0.7151522, 0.0721750],
                               [0.0193339, 0.1191920, 0.9503041]])

    M_XYZ_to_cam = camera_matrix / 10000.0
    M_XYZ_to_cam = M_XYZ_to_cam.reshape(3, 3)

    # Compute the sRGB to camera RGB matrix
    M_sRGB_to_cam = np.dot(M_XYZ_to_cam, M_sRGB_to_XYZ)

    # Normalize the rows of the matrix
    M_sRGB_to_cam /= M_sRGB_to_cam.sum(axis=1, keepdims=True)

    # Compute the inverse matrix
    M_cam_to_sRGB = np.linalg.inv(M_sRGB_to_cam)

    # Apply the color space correction
    demosaiced_img_sRGB = np.zeros_like(demosaiced_img)
    for y in range(demosaiced_img.shape[0]):
        for x in range(demosaiced_img.shape[1]):
            demosaiced_img_sRGB[y, x] = np.dot(M_cam_to_sRGB, demosaiced_img[y, x])

    return np.clip(demosaiced_img_sRGB, 0, 1)

# Camera-specific matrix from dcraw source code for Nikon D3400
# { 6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318 }
camera_matrix = np.array([6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318])
demosaiced_white_world_img_sRGB = color_space_correction(demosaic_bilinear(white_world_img, 'rggb'), camera_matrix)
demosaiced_gray_world_img_sRGB = color_space_correction(demosaic_bilinear(gray_world_img, 'rggb'), camera_matrix)
demosaiced_camera_presets_img_sRGB = color_space_correction(demosaic_bilinear(camera_presets_img, 'rggb'), camera_matrix)

plt.imsave('data/white_world/color_space_correction_and_demosaicing/image.png', demosaiced_white_world_img_sRGB)
plt.imsave('data/gray_world/color_space_correction_and_demosaicing/image.png', demosaiced_gray_world_img_sRGB)
plt.imsave('data/camera_presets/color_space_correction_and_demosaicing/image.png', demosaiced_camera_presets_img_sRGB)


# ------------------ BRIGHTNESS ADJUSTMENT AND GAMMA ENCODING ------------------

def gamma_encoding(demosaiced_img_sRGB, desired_mean=0.25):
    grayscale_img = rgb2gray(demosaiced_img_sRGB)
    mean_intensity = np.mean(grayscale_img)

    scaled_img = demosaiced_img_sRGB * (desired_mean / mean_intensity)
    clipped_img = np.clip(scaled_img, 0, 1)

    gamma_encoded_img = np.where(clipped_img <= 0.0031308,
                                 12.92 * clipped_img,
                                 (1 + 0.055) * np.power(clipped_img, 1 / 2.4) - 0.055)

    # Ensure the gamma_encoded_img is treated as an RGB image
    gamma_encoded_img = np.dstack([gamma_encoded_img[:, :, 0],
                                   gamma_encoded_img[:, :, 1],
                                   gamma_encoded_img[:, :, 2]])

    return gamma_encoded_img

gamma_white_world = gamma_encoding(demosaiced_white_world_img_sRGB)
gamma_gray_world = gamma_encoding(demosaiced_gray_world_img_sRGB)
gamma_camera_presets = gamma_encoding(demosaiced_camera_presets_img_sRGB)
matplotlib.use("MacOSX")

plt.imsave('data/white_world/gamma_encoding/image.png', gamma_white_world)
plt.imsave('data/gray_world/gamma_encoding/image.png', gamma_gray_world)
plt.imsave('data/camera_presets/gamma_encoding/image.png', gamma_camera_presets)


# ------------------ COMPRESSION ------------------

def compress_and_save_image(gamma_encoded_img, output_dir):
    # Save as uncompressed PNG
    matplotlib.use("MacOSX")
    png_file = os.path.join(output_dir, 'image.png')
    plt.imsave(png_file, (gamma_encoded_img * 255).astype(np.uint8), cmap='gray')
    png_size = os.path.getsize(png_file)

    # Save as JPEG with quality=95
    jpeg_file_q95 = os.path.join(output_dir, 'image_q95.jpeg')
    io.imsave(jpeg_file_q95, (gamma_encoded_img * 255).astype(np.uint8), quality=95)
    jpeg_size_q95 = os.path.getsize(jpeg_file_q95)

    # Compute compression ratio for JPEG with quality=95
    compression_ratio_q95 = png_size / jpeg_size_q95
    #print(f"Compression ratio for JPEG with quality=95: {compression_ratio_q95:.2f}")

    # Find the lowest JPEG quality setting with indistinguishable quality
    quality = 100
    while quality > 0:
        jpeg_file = os.path.join(output_dir, f'image_q{quality}.jpeg')
        io.imsave(jpeg_file, (gamma_encoded_img * 255).astype(np.uint8), quality=quality)
        jpeg_size = os.path.getsize(jpeg_file)

        if jpeg_size < png_size:
            compression_ratio = png_size / jpeg_size
            print(f"Compression ratio for JPEG with quality={quality}: {compression_ratio:.2f}")
            break

        quality -= 5

compress_and_save_image(gamma_white_world, 'data/white_world/compression')
compress_and_save_image(gamma_gray_world, 'data/gray_world/compression')
compress_and_save_image(gamma_camera_presets, 'data/camera_presets/compression')


# ------------------ MANUAL WHITE BALANCING ------------------

def manual_white_balancing(image_path, white_patch_coords, brightness_coeff, folder_path):
    img = io.imread(image_path)
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Extract the white patch and calculate the mean values for each channel
    x, y, width, height = white_patch_coords
    white_patch = img[y:y+height, x:x+width]
    mean_values = np.mean(white_patch, axis=(0, 1))
    
    # Calculate scaling factors to normalize the RGB channels and apply scaling vectors
    scale_factors = mean_values.max() / mean_values
    white_balanced_image = (img * scale_factors).clip(0, 1)
    
    # Save the original images
    descriptive_name = os.path.basename(os.path.dirname(image_path))
    original_image_save_path = os.path.join(folder_path, f'{descriptive_name}_orig.png')
    plt.imsave(original_image_save_path, img)
    
    # Save the white-balanced image
    white_balanced_image_save_path = os.path.join(folder_path, f'{descriptive_name}_wb.png')
    white_balanced_image *= brightness_coeff
    plt.imsave(white_balanced_image_save_path, white_balanced_image)


def get_white_patch_coords(image_path):
    img = io.imread(image_path)
    
    # Convert the image to double-precision array and normalize
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Display the image and use ginput to select the white patch
    plt.imshow(img)
    plt.title('Click on the top left corner and bottom right corner of the desired white patch')
    plt.axis('on')
    points = plt.ginput(2)
    plt.close()
    
    # Calculate the size of the selected white patch
    x1, y1 = points[0]
    x2, y2 = points[1]
    minx, miny = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)
    white_patch_coords = (int(minx), int(miny), int(width), int(height))
    return white_patch_coords

white_patch_coords = get_white_patch_coords('data/baby.jpeg')
manual_white_balancing('data/white_world/compression/image.png', white_patch_coords, 0.775, 'data/white_world/final_images')
manual_white_balancing('data/gray_world/compression/image.png', white_patch_coords, 0.775, 'data/gray_world/final_images')
manual_white_balancing('data/camera_presets/compression/image.png', white_patch_coords, 0.775, 'data/camera_presets/final_images')
