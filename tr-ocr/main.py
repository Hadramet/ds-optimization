# tr-ocr fastapi for inference
from fastapi import FastAPI, File, UploadFile
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

app = FastAPI()

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')


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
    image = Image.open(file.file)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]    
    return {"text": generated_text}

# Command to run the server locally
# uvicorn main:app --reload --host
