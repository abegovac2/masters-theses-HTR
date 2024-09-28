import xml.etree.ElementTree as ET
import os


def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id, classes, input_dir, output_dir):
    in_file = open(f"{input_dir}/{image_id}.xml", "r", encoding="utf8")
    out_file = open(f"{output_dir}/{image_id}.txt", "w", encoding="utf8")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        # cls = obj.find('name').text
        cls = "word"
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )
        bb = convert_bbox((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")

    in_file.close()
    out_file.close()


def pascalVOC2yolo(file_name, src, dest=None):
    if dest is None:
        dest = src
    convert_annotation(file_name, ["word"], src, dest)
