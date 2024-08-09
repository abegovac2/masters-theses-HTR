from typing import Tuple
from rest.config import settings
from rest.clients.groq_client import GroqClient
from rest.services import (
    RegionExtractionService,
    WordExtractionService,
    TextExtractionService,
)


def get_extraction_services() -> (
    Tuple[RegionExtractionService, WordExtractionService, TextExtractionService]
):
    return (
        RegionExtractionService(model_path=settings.REGION_EXTRACTION_MODEL_PATH),
        WordExtractionService(model_path=settings.WORD_EXTRACTION_MODEL_PATH),
        TextExtractionService(model_path=settings.TEXT_EXTRACTION_MODEL_PATH),
    )


def get_groq_client() -> GroqClient:
    return GroqClient(
        settings.GROQ_MODEL, settings.GROQ_API_KEYS, settings.FUEL_SHOT_SIZE
    )
