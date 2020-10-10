FROM jjanzic/docker-python3-opencv
WORKDIR .
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy
RUN pip install pandas
RUN pip install Cython
RUN pip install pycocotools

FROM anibali/pytorch:1.4.0-cuda10.1
RUN sudo apt update && sudo apt install -y build-essential
WORKDIR .
COPY requirements.txt .
RUN pip install --upgrade Pillow
RUN pip install -r requirements.txt
