from ultralytics import YOLO
from rest.models.model import Document, Region, BoundingBox, Point, Image
from utils.extract_regions import extract_rectangle


class RegionExtractionService:
    model = None

    def __init__(self, model_path="ml_models/region_detection_model.pt") -> None:
        if self.model is None:
            self.model = YOLO(model_path)

    def extract(self, document: Document):
        image = document.input_image.image
        results = self.model.predict(source=image, imgsz=640, conf=0.25, iou=0.45)
        results = results[0]
        boxes = results.boxes
        classes = boxes.cls
        zipped = list(zip(classes, boxes))
        regions = []
        classes = []
        for cls, box in zipped:
            tensor = box.xyxy[0]
            x1 = int(tensor[0].item())
            y1 = int(tensor[1].item())
            x2 = int(tensor[2].item())
            y2 = int(tensor[3].item())
            regions.append([[x1, y1], [x2, y2]])
            classes.append(int(cls))

        document.regions = {
            c: Region(
                region_image=Image(
                    image=extract_rectangle(image, r[0], r[1], wiggle_room=15),
                    title=f"{document.input_image.title}_{c}",
                ),
                bounding_box=BoundingBox(
                    top_left=Point(x=r[0][0], y=r[0][1]),
                    bottom_right=Point(x=r[1][0], y=r[1][1]),
                ),
            )
            for c, r in list(zip(classes, regions))
        }

        return document
