from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from cv2.typing import MatLike
import cv2
import numpy as np
import base64


def numpy_array_serializer(img: np.ndarray):
    if not img.any():
        return b""
    return base64.b64encode(img.tobytes())


def cv2_image_serializer(img: MatLike):
    if not img.any():
        return b""
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()


class Raw(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: numpy_array_serializer,
            MatLike: cv2_image_serializer,
        }


class Image(Raw):
    image: Union[str, MatLike, np.ndarray]
    format: str = "jpg"
    title: Optional[str]


class Point(Raw):
    x: int
    y: int


class BoundingBox(Raw):
    top_left: Point
    bottom_right: Point


class Detection(Raw):
    text: str
    line_image: Image
    probability: float
    bounding_box: BoundingBox


class Region(Raw):
    region_image: Image
    bounding_box: BoundingBox
    detections: List[Detection] = []


class Document(Raw):
    input_image: Image
    regions: Dict[int, Region] = {}
