import os
import xml.etree.ElementTree as ET


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    filename = root.find("filename").text

    size_node = root.find("size")
    h = int(size_node.find("height").text)
    w = int(size_node.find("width").text)

    for w_idx, boxes in enumerate(root.iter("object")):
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        name = str(boxes.find("name").text).upper()

        row = [filename, w, h, xmin, ymin, xmax, ymax, w_idx, name]
        list_with_all_boxes.append(row)

    return list_with_all_boxes


def generate_word_labels(path, labels):
    """
    Format
    filename, img_width, img_height, xmin, ymin, xmax, ymax, word_idx, word
    """
    with open(f"{path}/word_labels.txt", "w", encoding="utf8") as file:
        for label in labels:
            xml_file = f"{path}/{label}"
            boxes = read_content(xml_file)
            label = label.removesuffix(".xml")
            for box in boxes:
                b = [str(i) for i in box]
                file.write(f'{"|".join(b)}\n')


from sklearn.cluster import DBSCAN


def generate_line_labels(path, labels):
    """
    Format
    filename, img_width, img_height, line_idx, line_x_min, line_y_min, line_x_max, line_y_max, text
    """
    with open(f"{path}/line_labels.txt", "w", encoding="utf8") as file:
        for label in labels:
            xml_file = f"{path}/{label}"
            boxes = read_content(xml_file)
            label = label.removesuffix(".xml")
            w0, h0 = boxes[0][1], boxes[0][2]
            filename0 = boxes[0][0]
            per_line = h0 // 11 + 2
            regions = [((i + 1) * per_line, []) for i in range(11)]
            regions.append((h0, []))
            middle = [(box[6] + box[4]) // 2 for box in boxes]
            distances = [[abs(i - j) for j in middle] for i in middle]
            cluster = DBSCAN(eps=25, min_samples=1, metric="precomputed").fit(distances)
            clusters = [i for i in cluster.labels_]
            regions = {}
            for clust, box in list(zip(clusters, boxes)):
                if regions.get(clust, None) is None:
                    regions[clust] = [box]
                else:
                    regions[clust].append(box)

            for l_idx, (clust, boxes) in enumerate(regions.items()):
                boxes = list(sorted(boxes, key=lambda box: box[3]))
                text = " ".join([box[8] for box in boxes])
                text = text.strip(" ")
                if len(text) == 0:
                    continue
                l_x_min = min([box[3] for box in boxes])
                l_y_min = min([box[4] for box in boxes])
                l_x_max = max([box[5] for box in boxes])
                l_y_max = max([box[6] for box in boxes])
                line = [
                    filename0,
                    w0,
                    h0,
                    l_idx,
                    l_x_min,
                    l_y_min,
                    l_x_max,
                    l_y_max,
                    text,
                ]
                line = "|".join([str(el) for el in line])
                file.write(f"{line}\n")


def generate_region_labels(path, labels):
    """
    Format
    filename, img_width, img_height, region_idx, region_y_min, region_y_max, text
    """
    with open(f"{path}/region_labels.txt", "w", encoding="utf8") as file:
        for label in labels:
            xml_file = f"{path}/{label}"
            boxes = read_content(xml_file)
            label = label.removesuffix(".xml")
            w0, h0 = boxes[0][1], boxes[0][2]
            filename0 = boxes[0][0]
            region_sizes = [3, 3, 3, 2]
            s = sum(region_sizes) + 2
            per_line = h0 // s
            regions = {
                r * (i + 1) * per_line: [] for i, r in enumerate(region_sizes[:-1])
            }
            regions[h0] = []
            for box in boxes:
                for dims, boxes1 in regions.items():
                    if box[4] < dims:
                        boxes1.append(box)
                        break

            regions = list(regions.items())

            for r_idx, (dims, boxes) in enumerate(regions):
                text = " ".join([box[8] for box in boxes])
                text = text.strip(" ")
                if len(text) == 0:
                    continue
                r_y_min = 0 if r_idx == 0 else regions[r_idx - 1][0]
                r_y_max = dims
                region = [filename0, w0, h0, r_idx, r_y_min, r_y_max, text]
                region = "|".join([str(el) for el in region])
                file.write(f"{region}\n")
