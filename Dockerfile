FROM anibali/pytorch:cuda-10.0

USER root
RUN mkdir -p /home/app/uploads
RUN pip install --upgrade pip setuptools wheel
RUN sudo apt update && sudo apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg vim
RUN mkdir -p outputs

COPY requirements.txt .

RUN pip install -r requirements.txt
WORKDIR /home/app/

COPY ./ ./

CMD ["python", "main.py"]