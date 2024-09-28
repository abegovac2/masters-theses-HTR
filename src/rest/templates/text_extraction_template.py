import asyncio
import numpy as np
from rest.services import (
    TextExtractionService,
    WordExtractionService,
    RegionExtractionService,
)
from rest.clients.groq_client import GroqClient

from rest.models.model import Document, Detection, Region, BoundingBox, Point, Image
from utils.extract_regions import extract_rectangle
from typing import List, Dict
from sklearn.cluster import DBSCAN
from statistics import mean


class TextExtractionTemplate:
    def __init__(
        self,
        text_extraction_service: TextExtractionService,
        word_extraction_service: WordExtractionService,
        region_extraction_service: RegionExtractionService,
        groq_client: GroqClient,
        groq_threshold: float = 0.5,
    ) -> None:
        self._tes = text_extraction_service
        self._wes = word_extraction_service
        self._res = region_extraction_service
        self._client = groq_client
        self._groq_threshold = groq_threshold

    def extract_regions(self, document: Document) -> Document:
        return self._res.extract(document)

    def extract_words(self, document: Document) -> Document:
        regions = [val for key, val in document.regions.items()]

        for region in regions:
            self._wes.extract(region)
            # region.region_image.image = np.ndarray([0, 0])

        return document

    def extract_text(self, document: Document) -> Document:
        for region_idx, region in document.regions.items():
            for detection in region.detections:
                self._tes.extract(detection)

        return document

    def join_detections_into_lines(self, document: Document) -> Document:
        for region_idx, region in document.regions.items():
            detections = region.detections
            midpoints = [
                (
                    detection.bounding_box.bottom_right.y
                    + detection.bounding_box.top_left.y
                )
                // 2
                for detection in detections
            ]
            distances = [[abs(i - j) for j in midpoints] for i in midpoints]
            cluster = DBSCAN(eps=25, min_samples=1, metric="precomputed").fit(distances)
            clusters = [i for i in cluster.labels_]

            regions: Dict[int, List[Detection]] = {}
            for clust, detection in list(zip(clusters, detections)):
                if regions.get(clust, None) is None:
                    regions[clust] = [detection]
                    detection.bounding_box.top_left.x
                else:
                    regions[clust].append(detection)

            regions = {
                key: list(sorted(value, key=lambda dets: dets.bounding_box.top_left.x))
                for key, value in regions.items()
            }

            merged_detections: List[Detection] = []
            for cluster, boxes in regions.items():
                boxes = list(sorted(boxes, key=lambda det: det.bounding_box.top_left.x))

                x_min = min([box.bounding_box.top_left.x for box in boxes])
                y_min = min([box.bounding_box.top_left.y for box in boxes])
                x_max = max([box.bounding_box.bottom_right.x for box in boxes])
                y_max = max([box.bounding_box.bottom_right.y for box in boxes])

                merged_detections.append(
                    Detection(
                        text=" ".join([box.text for box in boxes]),
                        probability=mean([box.probability for box in boxes]),
                        line_image=Image(
                            image=extract_rectangle(
                                region.region_image.image,
                                (x_min, y_min),
                                (x_max, y_max),
                            ),
                            title=f"{region.region_image.title}_{cluster}",
                        ),
                        bounding_box=BoundingBox(
                            top_left=Point(x=x_min, y=y_min),
                            bottom_right=Point(x=x_max, y=y_max),
                        ),
                    )
                )

            region.detections = merged_detections

        return document

    async def enhance_with_llm(self, document: Document) -> Document:
        async_jobs = [
            (
                (
                    asyncio.create_task(self._client.correct_extraction(detection.text))
                    if detection.probability < self._groq_threshold
                    else detection.text
                ),
                region_idx,
                detecion_idx,
            )
            for region_idx, region in document.regions.items()
            for detecion_idx, detection in enumerate(region.detections)
        ]

        await asyncio.gather(
            *[job[0] for job in async_jobs if not isinstance(job[0], str)]
        )

        for text, region_idx, detection_idx in async_jobs:
            if not isinstance(text, str):
                detection = document.regions[region_idx].detections[detection_idx]
                detection.text = text.result()
                detection.probability = -1.0

        return document
