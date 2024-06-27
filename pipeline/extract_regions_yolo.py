from ultralytics import YOLO
import cv2
from utils.extract_regions import extract_rectangle


class ExtractDocumentRegions:
    model = None

    def __init__(self, model_path="./models/region_detection_model.pt") -> None:
        if self.model is None:
            self.model = YOLO(model_path)

    def _extract_interesting_regions(self, image):
        results = self.model.predict(source=image, imgsz=640, conf=0.25, iou=0.45)
        results = results[0]
        boxes = results.boxes
        classes = boxes.cls
        zipped = list(zip(classes, boxes))
        rectangles = []
        classes = []
        for cls, box in zipped:
            tensor = box.xyxy[0]
            x1 = int(tensor[0].item())
            y1 = int(tensor[1].item())
            x2 = int(tensor[2].item())
            y2 = int(tensor[3].item())
            rectangles.append([[x1, y1], [x2, y2]])
            classes.append(int(cls))
        return rectangles, classes

    def execute(self, image, interest_region):
        if isinstance(interest_region, int):
            interest_region = [interest_region]

        regions, classes = self._extract_interesting_regions(image)

        cls_dict = {c: r for c, r in list(zip(classes, regions))}

        regions = {r: cls_dict.get(r, None) for r in interest_region}
        regions = [
            [key, extract_rectangle(image, region[0], region[1], wiggle_room=15)]
            for key, region in regions.items()
            if region is not None
        ]

        return regions
