# tr-ocr fastapi for inference
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

@app.on_event("startup")
async def startup_event():
    print("Loading model...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down model...")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    image = Image.open(file.file).convert('RGB')
    result = ocr.ocr(np.array(image), det=False, rec=True, cls=False)
    text = '\n'.join([i[0][0] for i in result])
    confidence = '\n'.join([str(i[0][1]) for i in result])
    return {"text": text, "confidence": confidence}