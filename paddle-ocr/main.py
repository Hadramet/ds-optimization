# paddle ocr fastapi for inference

import os
import sys
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from paddleocr import PaddleOCR
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime

# Define the FastAPI app
app = FastAPI()

# Define the model
class Model(BaseModel):
    name: str
    version: str
    date: str
    description: str
    url: str

# Define the prediction
class Prediction(BaseModel):
    label: str
    confidence: float


# app startup event
@app.on_event("startup")
async def startup_event():
    global paddle_ocr
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")

# app shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    paddle_ocr = None

# Define the root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Define the inference endpoint
@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Inference
    prediction = paddle_ocr.ocr(np.array(image))

    # Format the prediction
    prediction = [Prediction(label=p[1][0], confidence=p[1][1]) for p in prediction[0]]

    return prediction
