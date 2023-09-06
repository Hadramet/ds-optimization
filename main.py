# Ensemple prediction
# For each sample in your dataset, obtain predictions from each of the pretrained models: TrOCR, PaddleOCR, and KerasOCR.

import os
import sys
import json
import requests

def trOcrInference(image) -> tuple[str, float]:
    tr_ocr_url = "http://localhost:8086/inference"
    files = {'file': open(image, 'rb')}
    response = requests.post(tr_ocr_url, files=files)
    response_json = response.json()
    text = response_json['text']
    conf = 0.0
    print(f"\ttrOcrInference: text: {text}, conf: {conf}")
    return text, conf



def paddleOcrInference(image) -> tuple[str, float]:
    return "dummy", 0.0

def kerasOcrInference(image) -> tuple[str, float]:
    return "dummy", 0.0

def ensemblePediction(images_txt: str):
    assert os.path.exists(images_txt), f"File not found: {images_txt}"
    with open(images_txt, "r") as f:
        images = f.readlines()

    # images_txt_file = os.path.basename(images_txt)
    images_txt_folder = os.path.dirname(images_txt)

    ensemble = {}
    for image in images[:10]:
        line_split = image.split()
        image = line_split[0]
        label = " ".join(line_split[1:])
        print(f"image: {image}, label: {label}")
        image = image[1:]
        image_full_path = os.path.join(images_txt_folder, image)
        trOcrPred, trOcrConf = trOcrInference(image_full_path)
        paddleOcrPred, paddleOcrConf = paddleOcrInference(image_full_path)
        kerasOcrPred, kerasOcrConf = kerasOcrInference(image_full_path)

        ensemble[image] = {'label': label}
        ensemble[image]['trOcr'] = {'pred': trOcrPred, 'conf': trOcrConf}
        ensemble[image]['paddleOcr'] = {'pred': paddleOcrPred, 'conf': paddleOcrConf}
        ensemble[image]['kerasOcr'] = {'pred': kerasOcrPred, 'conf': kerasOcrConf}

    output_file = "ensemble.json" 
    output_file = os.path.join(images_txt_folder, output_file)
    
    with open(output_file, "w") as f:
        json.dump(ensemble, f)

def consensus_voting(ensemble: dict) -> dict:
    pass


def main():
    input_file = sys.argv[1]
    ensemblePediction(input_file)

if __name__ == "__main__":
    main()
