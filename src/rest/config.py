from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    REGION_EXTRACTION_MODEL_PATH: str = "./ml_models/region_detection_model.pt"
    WORD_EXTRACTION_MODEL_PATH: str = "./ml_models/word_detection_model.pt"
    TEXT_EXTRACTION_MODEL_PATH: str = (
        "./ml_models/mine_logs_v4/flor/text_detection_model_v4.hdf5"
    )
    GROQ_API_KEYS: List[str] = []
    GROQ_MODEL: str = "llama3-8b-8192"
    FEW_SHOT_SIZE: int = 50
    GROQ_THRESHOLD: float = 0.5

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
