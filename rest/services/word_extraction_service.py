from ultralytics import YOLO
from rest.models.model import Region, Detection, BoundingBox, Point, Image
from sklearn.cluster import DBSCAN
from utils.extract_regions import extract_rectangle


class WordExtractionService:
    model = None

    def __init__(self, model_path) -> None:
        if self.model is None:
            self.model = YOLO(model_path)

    def _extract_word_bnbx(self, image):
        results = self.model.predict(source=image, imgsz=640, conf=0.25, iou=0.45)
        results = results[0]
        boxes = results.boxes
        rectangles = []
        for box in boxes:
            tensor = box.xyxy[0]
            x1 = int(tensor[0].item())
            y1 = int(tensor[1].item())
            x2 = int(tensor[2].item())
            y2 = int(tensor[3].item())
            rectangles.append([[x1, y1], [x2, y2]])
        return rectangles

    def extract(self, region: Region):
        image = region.region_image.image
        word_rect = self._extract_word_bnbx(image)
        if len(word_rect) == 0:
            return []

        region.detections = [
            Detection(
                text="",
                probability=0,
                line_image=Image(
                    image=extract_rectangle(image, word[0], word[1]),
                    title=f"{region.region_image.title}_{idx}",
                ),
                bounding_box=BoundingBox(
                    top_left=Point(x=word[0][0], y=word[0][1]),
                    bottom_right=Point(x=word[1][0], y=word[1][1]),
                ),
            )
            for idx, word in enumerate(word_rect)
        ]

        region.detections = list(
            sorted(region.detections, key=lambda det: det.bounding_box.top_left.y)
        )

        return region.detections
