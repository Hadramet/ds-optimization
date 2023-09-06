# keras ocr fastapi for inference

import os
import sys
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from keras_ocr import pipeline
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
    global keras_ocr_pipeline
    keras_ocr_pipeline = pipeline.Pipeline()

# app shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    keras_ocr_pipeline = None

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
    prediction = keras_ocr_pipeline.recognize([np.array(image)])[0]

    # Format the prediction
    predictions = []
    for pred in prediction:
        label = pred[0]
        confidence = pred[1]
        predictions.append(Prediction(label=label, confidence=confidence))

    return predictions
