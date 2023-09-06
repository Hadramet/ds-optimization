FROM continuumio/miniconda3

RUN mkdir /tr-ocr
WORKDIR /tr-ocr

COPY . .

RUN echo "Running $(conda --version)" && \
    pip install -r requirements.txt

EXPOSE 8086
CMD ["sh", "-c","uvicorn main:app --port 8086 --host 0.0.0.0 --reload"]

# docker build -t tr-ocr .
# docker run -p 8086:8086 tr-ocr 