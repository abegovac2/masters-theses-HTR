from .shared import *


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

    eroded_image = erode_image(thresholded_image)
    filtered_image = filter_small_components(eroded_image, min_area)

    clean_image = dilate_image(filtered_image)
    return clean_image


def apply_image_cleaning_pipeline(
    src: str, dest: str, threshold_value=127, min_area=100, n_chunks=4, save=True
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

    if save:
        cv2.imwrite(dest, output_image)

    return output_image
