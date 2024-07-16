from rest.services import (
    RegionExtractionService,
    WordExtractionService,
    TextExtractionService,
)
from rest.models import Document, Image, Detection
import cv2
from typing import List
import numpy as np

from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/v1/text-extraction")


@router.post("")
async def read_text_from_image(
    upload_image: UploadFile = File(media_type="image/jpeg"),
):
    title = upload_image.filename.split(".")[0]
    image = None
    with open("text.txt", "w") as f:
        f.write(f"{upload_image.filename}\n")
        image_bytes = upload_image.file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        f.write(f"{image.shape}\n")

    a = RegionExtractionService()
    b = WordExtractionService()
    c = TextExtractionService()

    doc = Document(input_image=Image(image=image, title=title))
    doc = a.extract(doc)
    regions = [val for key, val in doc.regions.items()]
    detections: List[Detection] = []

    for region in regions:
        detections.extend(b.extract(region))
        region.region_image.image = []

    for det in detections:
        c.extract(det)
        det.line_image.image = []

    doc.input_image.image = []

    return doc
    return doc.model_dump_json(indent=4)
