import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_difference(image1, image2):
    # Compute the absolute difference between the two images
    diff_image = cv2.absdiff(image1, image2)
    return diff_image


def compute_binary_mask(dif_single_channel):
    # # Convert the difference image to a single channel
    # gray_diff = cv2.cvtColor(dif_single_channel, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to the difference image
    _, binary_mask = cv2.threshold(
        dif_single_channel, 0, 255, cv2.THRESH_BINARY)
    return binary_mask


def replace_background(bg1_image, bg2_image, ob_image):
    # Compute the difference
    difference_single_channel = compute_difference(bg1_image, ob_image)
    # Compute the binary mask
    binary_mask = compute_binary_mask(difference_single_channel)
    # Replace the background
    output = np.where(binary_mask == 255, ob_image, bg2_image)
    return output


if __name__ == '__main__':
    # Compute the difference between the two images
    # Read and resize the images
    bg1_image = cv2.imread('Images/GreenBackground.png', 1)
    bg1_image = cv2.cvtColor(bg1_image, cv2.COLOR_BGR2RGB)
    bg1_image = cv2.resize(bg1_image, (678, 381))

    ob_image = cv2.imread('Images/Object.png', 1)
    ob_image = cv2.cvtColor(ob_image, cv2.COLOR_BGR2RGB)
    ob_image = cv2.resize(ob_image, (678, 381))

    bg2_image = cv2.imread('Images/NewBackground.jpg', 1)
    bg2_image = cv2.cvtColor(bg2_image, cv2.COLOR_BGR2RGB)
    bg2_image = cv2.resize(bg2_image, (678, 381))

    diff_image = compute_difference(bg1_image, ob_image)
    binary_mask = compute_binary_mask(diff_image)
    result = replace_background(bg1_image, bg2_image, ob_image)
    # Display the difference image
    plt.imshow(diff_image)
    plt.imshow(binary_mask)
    plt.imshow(result)
    plt.show()
