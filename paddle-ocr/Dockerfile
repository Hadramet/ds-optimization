FROM ubuntu:20.04
# use continuumio/miniconda3 as base image
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /paddle-ocr
WORKDIR /paddle-ocr

COPY . .
COPY requirements.txt /tmp

RUN apt-get update \
    && apt-get install -y python3 python3-pip libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && pip install -r /tmp/requirements.txt

RUN apt-get -y remove python3-pip \
    && apt-get -y autoremove \
    && apt-get -y install --no-install-recommends python3-setuptools \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/ /root/.cache /tmp/*

EXPOSE 8087
CMD ["sh", "-c","uvicorn main:app --port 8087 --host 0.0.0.0 --reload"]
# docker build -t tr-ocr .
# docker run -p 8080:8087 tr-ocr 
