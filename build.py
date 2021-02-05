import boto3
import os
from tqdm import tqdm
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AWSAccessKeyId = os.environ.get("AWSAccessKeyId")
AWSSecretKey = os.environ.get("AWSSecretKey")
ROOT_DIR = f"{SCRIPT_DIR}/"
BUCKET_NAME = os.environ.get("BUCKET_NAME")

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
        aws_access_key_id=AWSAccessKeyId,
        aws_secret_access_key=AWSSecretKey,
    )

    s3_client = session.resource('s3')

    for file, directory in tqdm(files_directory.items()):
        file_name = os.path.join(ROOT_DIR, directory, file)
        if not os.path.exists(file_name):
            s3_client.Bucket(BUCKET_NAME).download_file(file, file_name)
            logging.debug(f"File {file} successfully downloaded")
        else:
            logging.debug(f"File {file} already exists")


if __name__ == "__main__":
    build()

