from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    REGION_EXTRACTION_MODEL_PATH: str = "ml_models/region_detection_model.pt"
    WORD_EXTRACTION_MODEL_PATH: str = "ml_models/word_detection_model.pt"
    TEXT_EXTRACTION_MODEL_PATH: str = "ml_models/text_detection_model.hdf5"


settings = Settings()
