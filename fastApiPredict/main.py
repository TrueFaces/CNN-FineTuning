import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import Depends, FastAPI, UploadFile
from pydantic import BaseModel

## Add Doc Security
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer
from fastapi.logger import logger
import logging
from app.internal import auth

from app.routers import images, users
from app.db.database import SessionLocal, engine, Base
from app.config import Settings

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Logs
uvicorn_logger = logging.getLogger('uvicorn.error')
logger.handlers = uvicorn_logger.handlers

# Dependency
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

app.include_router(auth.router)
app.include_router(users.router, dependencies=[Depends(oauth2_scheme)])
app.include_router(images.router, dependencies=[Depends(oauth2_scheme)])

@app.get("/docs", include_in_schema=False)
async def get_swagger_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation():
    return get_redoc_html(openapi_url="/openapi.json", title="docs")

@app.get("/")
async def root():
    return {"status": "runnig"}



class PredictionSchema(BaseModel):
    filename: str
    content_type: str
    prediction: str


# Carga del modelo de detección de caras
model = load_model("/app/modelo.h5")




# Ruta para realizar las predicciones a partir de un archivo
@app.post("/predict", response_model=PredictionSchema)
async def predict(file: UploadFile):
    # Cargar imagen y aplicar preprocesamiento
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # Hacer la predicción
    prediction = model.predict(img)

    # Devolver la predicción
    if prediction[0][0] > 0.5:
        face = "La imagen es una cara."
    else:
        face = "La imagen no es una cara."

    # Devolver la lista de caras detectadas
    
    return PredictionSchema(
        filename=file.filename,
        content_type=file.content_type,
        prediction=face
    )
   

