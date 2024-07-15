from ultralytics import YOLO
import cv2
from sklearn.cluster import DBSCAN
from utils.extract_regions import extract_rectangle


class ExtractWordsYolo:
    model = None

    def __init__(self, model_path="./models/word_detection_model.pt") -> None:
        if self.model is None:
            self.model = YOLO(model_path)

    def _extract_words(self, image):
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

        s = list(sorted(list(zip(rectangles, classes)), key=lambda rect: rect[0][0][1]))
        rectangles, classes = list(zip(*s))
        return rectangles, classes

    def execute(self, image):
        rectangles, _ = self._extract_words(image)
        middle = [(box[0][1] + box[1][1]) // 2 for box in rectangles]
        distances = [[abs(i - j) for j in middle] for i in middle]

        cluster = DBSCAN(eps=25, min_samples=1, metric="precomputed").fit(distances)
        clusters = [i for i in cluster.labels_]

        regions = {}
        for clust, box in list(zip(clusters, rectangles)):
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

        return [extract_rectangle(image, line[0], line[1]) for line in lines]
