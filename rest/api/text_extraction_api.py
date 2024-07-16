from rest.services import (
    RegionExtractionService,
    WordExtractionService,
    TextExtractionService,
)
from rest.models import Document, Image, Detection
from rest.config import settings
import cv2
from typing import List, Tuple
import numpy as np

from fastapi import APIRouter, UploadFile, File, Depends


router = APIRouter(prefix="/v1/text-extraction")


def build_extraction_services() -> (
    Tuple[RegionExtractionService, WordExtractionService, TextExtractionService]
):
    return (
        RegionExtractionService(model_path=settings.REGION_EXTRACTION_MODEL_PATH),
        WordExtractionService(model_path=settings.WORD_EXTRACTION_MODEL_PATH),
        TextExtractionService(model_path=settings.TEXT_EXTRACTION_MODEL_PATH),
    )


@router.post("")
async def read_text_from_image(
    upload_image: UploadFile = File(media_type="image/jpeg"),
    extraction_services=Depends(build_extraction_services),
):
    title = upload_image.filename.split(".")[0]
    image_bytes = upload_image.file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    res, wes, tes = extraction_services

    doc = Document(input_image=Image(image=image, title=title))
    doc = res.extract(doc)
    regions = [val for key, val in doc.regions.items()]
    detections: List[Detection] = []

    for region in regions:
        detections.extend(wes.extract(region))
        region.region_image.image = []

    for det in detections:
        tes.extract(det)
        det.line_image.image = []

    doc.input_image.image = []

    return doc
