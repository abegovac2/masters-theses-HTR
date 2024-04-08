import cv2
import numpy as np


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


def find_connected_components(image):
    """
    Uzima jedan pixel gleda koliko ima susjednih bijelih piksela (odredjeno sa connectivity)
    Ako ima tu cifru onda ga zabiljezi kao komponentu
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    return num_labels, labels, stats, centroids


def filter_small_components(image, min_area):
    image = 255 - image
    num_labels, labels, stats, centroids = find_connected_components(image)
    filtered_image = np.zeros_like(image)
    for label, stat in enumerate(stats[1:], start=1):
        if stat[cv2.CC_STAT_AREA] >= min_area:
            filtered_image[labels == label] = 255
    return 255 - filtered_image


def apply_pipeline(image, threshold_value=127, min_area=100):
    gray_image = convert_to_gray(image)

    thresholded_image = threshold_image(gray_image, threshold_value)

    kernel = np.ones((3, 3), np.uint8)
    erroded_image = cv2.erode(thresholded_image, kernel, iterations=1)
    filtered_image = filter_small_components(erroded_image, min_area)

    clean_image = cv2.dilate(
        src=filtered_image,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT,
        iterations=1,
    )
    return clean_image


def apply_image_cleaning_pipeline(
    src: str, dest: str, threshold_value=127, min_area=100, n_chunks=4
):
    input_image = cv2.imread(src)

    image_chunks = chunk_image(input_image, n_chunks)

    transformed_chunks = []
    for chunk in image_chunks:
        transformed_chunk = apply_pipeline(
            chunk, threshold_value=threshold_value, min_area=min_area
        )
        transformed_chunks.append(transformed_chunk)

    output_image = join_chunks(transformed_chunks)

    cv2.imwrite(dest, output_image)
