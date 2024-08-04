import asyncio
import cv2
from rest.models import Document, Image, Detection
from rest.config import settings
from typing import List
import numpy as np

from fastapi import APIRouter, UploadFile, File, Depends
from rest.dependencies import get_extraction_services, get_groq_client


router = APIRouter(prefix="/v1/text-extraction")


@router.post("")
async def read_text_from_image(
    upload_image: UploadFile = File(media_type="image/jpeg"),
    extraction_services=Depends(get_extraction_services),
    llm_client=Depends(get_groq_client),
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
        region.region_image.image = np.ndarray([0, 0])

    for det in detections:
        tes.extract(det)
        # det.line_image.image = []

    async_jobs = [
        (
            asyncio.create_task(llm_client.correct_extraction(det.text))
            if det.probability < settings.GROQ_THRESHOLD
            else det.text
        )
        for det in detections
    ]

    await asyncio.gather(*[job for job in async_jobs if not isinstance(job, str)])

    for text, det in list(zip(async_jobs, detections)):
        if not isinstance(text, str):
            det.text = text.result()
            det.probability = -1.0

    doc.input_image.image = np.ndarray([0, 0])

    return doc
