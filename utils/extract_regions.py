from utils.shared import *


def get_horizontal_and_vertical_contures(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 2))
    detect_horizontal = cv2.morphologyEx(
        close, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts1 = cnts[0] if len(cnts) == 2 else cnts[1]

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 35))
    detect_vertical = cv2.morphologyEx(
        close, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = cnts[0] if len(cnts) == 2 else cnts[1]

    return cnts1 + cnts2


def get_left_right_bounding(heights):
    heights = sorted(heights, key=lambda rect: rect[3], reverse=True)
    heights = heights[:2]
    return sorted(heights, key=lambda rect: rect[0])


def get_horizontal_linest(widths, line_number=10):
    widths = sorted(widths, key=lambda rect: rect[2], reverse=True)
    widths = widths[:line_number]
    widths = sorted(widths, key=lambda rect: rect[1])
    return widths[1:][:-1]


def get_left_right_bottom_top(heights):
    x, y, w, h = heights[0]
    p1 = (x + w // 2, y)
    p2 = (x + w // 2, y + h)
    left = [p1, p2]

    x, y, w, h = heights[1]
    p1 = (x + w // 2, y)
    p2 = (x + w // 2, y + h)
    right = [p1, p2]

    p1 = (left[0][0], left[0][1])
    p2 = (right[0][0], right[0][1])
    top = [p1, p2]

    p1 = (left[1][0], left[1][1])
    p2 = (right[1][0], right[1][1])
    bottom = [p1, p2]

    return left, right, top, bottom


def format_widths(left, right, rect):
    x, y, w, h = rect
    p1 = (left[0][0], y + h // 2)
    p2 = (right[0][0], y + h // 2)
    return [p1, p2]


def filter_close_segments(lines, limit=300):
    lines = sorted(lines, key=lambda points: points[0][1])
    filtered_lines = []
    for line in lines:
        filtered = [
            min(distance(line[0], inserted[0]), distance(line[1], inserted[1])) < limit
            for inserted in filtered_lines
        ]
        if not any(filtered):
            filtered_lines.append(line)

    return filtered_lines


def convert_widths_to_rectangles(widths):
    rectangles = []
    for i in range(1, len(widths)):
        a = widths[i - 1]
        b = widths[i]
        rectangles.append([a[0], b[1]])

    return rectangles


def extract_rectangle(image, top_left, bottom_right, wiggle_room=None):
    if wiggle_room is not None:
        w, h, _ = image.shape
        x1p, y1p = top_left
        x2p, y2p = bottom_right

        top_left = [
            x1p - wiggle_room if 0 <= x1p - wiggle_room <= w else x1p,
            y1p - wiggle_room if 0 <= y1p - wiggle_room <= h else y1p,
        ]
        bottom_right = [
            x2p + wiggle_room if 0 <= x2p + wiggle_room <= w else x2p,
            y2p + wiggle_room if 0 <= y2p + wiggle_room <= h else y2p,
        ]
    x1, y1 = top_left
    x2, y2 = bottom_right
    extracted_region = image[y1:y2, x1:x2]
    return extracted_region


def prepare_image(image):
    gray = convert_to_gray(image)

    eroded = erode_image(gray)

    return threshold_image(eroded, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


def get_second_region_rectangles(rectangle):
    x0, y0 = rectangle[0]
    x1, y1 = rectangle[1]

    x3 = 29 * (x0 + x1) // 50
    y3 = (y0 + y1) // 2

    r2_1 = [(x0, y0), (x3, y1)]
    r2_2 = [(x3, y0), (x1, y3)]
    r2_3 = [(x3, y3), (x1, y1)]
    return [r2_1, r2_2, r2_3]


def get_fourth_region_rectangles(rectangle):
    x0, y0 = rectangle[0]
    x1, y1 = rectangle[1]

    r4_1 = [(x0, y0), ((x0 + x1) // 2, y1)]
    r4_2 = [((x0 + x1) // 2, y0), (x1, y1)]
    return [r4_1, r4_2]


def extract_image_regions(image):

    cnts = get_horizontal_and_vertical_contures(image)

    rectangles = [cv2.boundingRect(c) for c in cnts]

    heights = get_left_right_bounding(rectangles.copy())
    widths = get_horizontal_linest(rectangles)

    left, right, bottom, top = get_left_right_bottom_top(heights)

    widths = [format_widths(left, right, rect) for rect in widths]
    widths = [top, bottom, *widths]
    widths = filter_close_segments(widths)

    rectangles = convert_widths_to_rectangles(widths)

    r2 = get_second_region_rectangles(rectangles[1])
    r4 = get_fourth_region_rectangles(rectangles[3])

    rectangles = rectangles[:1] + r2 + rectangles[2:3] + r4 + rectangles[4:]

    return rectangles


def extract_interesting_regions(image):
    original = image.copy()

    image = prepare_image(image)

    rectangles = extract_image_regions(image)

    return [extract_rectangle(original, rect[0], rect[1]) for rect in rectangles]
