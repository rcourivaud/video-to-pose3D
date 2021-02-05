import boto3
import os
from tqdm import tqdm
import logging


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

ROOT_DIR = f"{SCRIPT_DIR}/"
MODELS_BUCKET_NAME = os.environ.get("MODELS_BUCKET_NAME")

os.system(f"mkdir -p {ROOT_DIR}/checkpoint")
os.system(f"mkdir -p  {ROOT_DIR}/joints_detectors/hrnet/models/pytorch//pose_coco/")

files_directory = {
    "duc_se.pth": "./joints_detectors/Alphapose/models/sppe/",
    "yolov3-spp.weights": "./joints_detectors/Alphapose/models/yolo/",
    "yolov3.weights": "./joints_detectors/hrnet/lib/detector/yolo/",
    "pretrained_h36m_detectron_coco.bin": "./checkpoint/",
    "pose_coco/pose_hrnet_w48_384x288.pth": "./joints_detectors/hrnet/models/pytorch/",
    "pose_coco/pose_hrnet_w48_256x192.pth": "./joints_detectors/hrnet/models/pytorch/",
    "pose_coco/pose_hrnet_w32_384x288.pth": "./joints_detectors/hrnet/models/pytorch/",
    "pose_coco/pose_hrnet_w32_256x192.pth": "./joints_detectors/hrnet/models/pytorch/",
}


def build():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    s3_client = session.resource('s3')

    for file, directory in tqdm(files_directory.items()):
        logging.info(f"Start downloading {file}...")
        file_name = os.path.join(ROOT_DIR, directory, file)
        if not os.path.exists(file_name):
            s3_client.Bucket(MODELS_BUCKET_NAME).download_file(file, file_name)
            logging.info(f"File {file} successfully downloaded")
        else:
            logging.info(f"File {file} already exists")


if __name__ == "__main__":
    build()

