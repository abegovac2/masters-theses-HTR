from fastapi import FastAPI

from rest.api.text_extraction_api import router as te_api


app = FastAPI(debug=True)


app.include_router(te_api)
