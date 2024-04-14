import numpy as np
import cv2
import math


def chunk_image(image, n):
    height, width = image.shape[:2]
    chunk_height = height // n
    chunks = []
    for i in range(n):
        start_y = i * chunk_height
        end_y = start_y + chunk_height
        if i == n - 1:
            end_y = height
        chunk = image[start_y:end_y, :]
        chunks.append(chunk)
    return chunks


def join_chunks(chunks):
    return np.vstack(chunks)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshold_image(
    image, threshold_value=127, max_value=255, threshold_type=cv2.THRESH_BINARY
):
    _, thresholded = cv2.threshold(image, threshold_value, max_value, threshold_type)
    return thresholded


def erode_image(image, kernel_dims=(3, 3), iterations=1):
    kernel = np.ones(kernel_dims, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def dilate_image(image, kernel_dims=(3, 3), iterations=1):
    kernel = np.ones(kernel_dims, np.uint8)
    return cv2.dilate(
        src=image,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT,
        iterations=1,
    )


def distance(p1, p2):
    return math.sqrt(
        (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1])
    )
