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

        midpoint = [(box[0][1] + box[1][1]) // 2 for box in word_rect]
        distances = [[abs(i - j) for j in midpoint] for i in midpoint]
        with open("./a.txt", "w") as f:
            f.write(f"{len(distances)}\n\n\n")
            for d in distances:
                f.write(f"{len(d)}\n")
        cluster = DBSCAN(eps=25, min_samples=1, metric="precomputed").fit(distances)
        clusters = [i for i in cluster.labels_]

        regions = {}
        for clust, box in list(zip(clusters, word_rect)):
            if regions.get(clust, None) is None:
                regions[clust] = [box]
            else:
                regions[clust].append(box)

        lines = []
        for cluster, boxes in regions.items():
            boxes = list(sorted(boxes, key=lambda box: box[0][0]))
            l_x_min = min([box[0][0] for box in boxes])
            l_y_min = min([box[0][1] for box in boxes])
            l_x_max = max([box[1][0] for box in boxes])
            l_y_max = max([box[1][1] for box in boxes])

            lines.append([[l_x_min, l_y_min], [l_x_max, l_y_max]])

        region.detections = [
            Detection(
                text="",
                certanty=0,
                line_image=Image(
                    image=extract_rectangle(image, line[0], line[1]),
                    title=f"{region.region_image.title}_{idx}",
                ),
                bounding_box=BoundingBox(
                    top_left=Point(x=line[0][0], y=line[0][1]),
                    bottom_right=Point(x=line[1][0], y=line[1][1]),
                ),
            )
            for idx, line in enumerate(lines)
        ]

        return region.detections
