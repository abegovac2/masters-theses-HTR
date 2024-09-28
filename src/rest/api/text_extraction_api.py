import cv2
import numpy as np

from rest.models import Document, Image, Detection

from fastapi import APIRouter, UploadFile, File, Depends
from rest.dependencies import get_extraction_services, get_groq_client
from rest.templates.text_extraction_template import TextExtractionTemplate


router = APIRouter(prefix="/v1/text-extraction")


@router.post("/line")
async def line(
    upload_image: UploadFile = File(media_type="image/jpeg"),
    extraction_services=Depends(get_extraction_services),
    llm_client=Depends(get_groq_client),
):
    title = upload_image.filename.split(".")[0]
    image_bytes = upload_image.file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    res, wes, tes = extraction_services
    template = TextExtractionTemplate(tes, wes, res, llm_client)

    doc = Document(input_image=Image(image=image, title=title))
    doc = template.extract_regions(doc)
    doc = template.extract_words(doc)
    doc = template.join_detections_into_lines(doc)
    doc = template.extract_text(doc)
    # doc = await template.enhance_with_llm(doc)

    return doc


@router.post("/word/a")
async def word_a(
    upload_image: UploadFile = File(media_type="image/jpeg"),
    extraction_services=Depends(get_extraction_services),
    llm_client=Depends(get_groq_client),
):
    title = upload_image.filename.split(".")[0]
    image_bytes = upload_image.file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    res, wes, tes = extraction_services
    template = TextExtractionTemplate(tes, wes, res, llm_client)

    doc = Document(input_image=Image(image=image, title=title))
    doc = template.extract_regions(doc)
    doc = template.extract_words(doc)
    doc = template.extract_text(doc)
    doc = template.join_detections_into_lines(doc)
    doc = await template.enhance_with_llm(doc)

    return doc


@router.post("/word/b")
async def word_b(
    upload_image: UploadFile = File(media_type="image/jpeg"),
    extraction_services=Depends(get_extraction_services),
    llm_client=Depends(get_groq_client),
):
    title = upload_image.filename.split(".")[0]
    image_bytes = upload_image.file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    res, wes, tes = extraction_services
    template = TextExtractionTemplate(tes, wes, res, llm_client)

    doc = Document(input_image=Image(image=image, title=title))
    doc = template.extract_regions(doc)
    doc = template.extract_words(doc)
    doc = template.extract_text(doc)
    doc = await template.enhance_with_llm(doc)
    doc = template.join_detections_into_lines(doc)

    return doc
