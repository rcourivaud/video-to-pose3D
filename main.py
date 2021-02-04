import pika
import sys
import json
import logging
import time
import requests
import os

sys.path.append("video-to-pose3D")
sys.path.append("video-to-pose3D/joints_detectors/Alphapose")

from videopose import inference_video

RABBITMQ_USERNAME = os.environ.get("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD")

KEYPOINT_MAPPING = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_cheek": 3,
    "right_cheek": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def process_json_file(json_data):
    def _process_frame(keypoints):
        xs = [elt for e, elt in enumerate(keypoints) if e % 3 == 0]
        ys = [elt for e, elt in enumerate(keypoints) if e % 3 == 1]
        zs = [elt for e, elt in enumerate(keypoints) if e % 3 == 2]
        keypoints_mapped = {
            name: {
                "x": xs[keypoint_index],
                "y": ys[keypoint_index],
                "z": zs[keypoint_index] * 100,
            } for name, keypoint_index in KEYPOINT_MAPPING.items()
        }
        return keypoints_mapped

    processed_results = dict(
        poses=[_process_frame(frame_result["keypoints"]) for frame_result in json_data],
        frames=len(json_data)
    )
    return processed_results


def build_json_results(id_):
    with open("outputs/alpha_pose_{id}/alphapose-results.json".format(id=id_), 'r') as f:
        alpha_result = json.load(f)
    processed_result = process_json_file(alpha_result)
    with open("outputs/alpha_pose_{id}/alphapose-results_processed.json".format(id=id_), "w") as f:
        json.dump(processed_result, f)


def process_message(ch, model, properties, body):
    message = json.loads(body)
    job_id = message['job_id']
    try:
        inference_video('uploads/' + job_id + '.mp4', 'alpha_pose')
        build_json_results(job_id)
        requests.put('http://api:8008/jobs/' + job_id, json={'status': 'PROCESSED'})
    except Exception as e:

        requests.put('http://api:8008/jobs/' + job_id, json={'status': 'FAILED'})
        raise e


def main():
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq',
                                                                           5672,
                                                                           '/',
                                                                           credentials,
                                                                           heartbeat=6000,
                                                                           blocked_connection_timeout=3000))
            channel = connection.channel()
            channel.queue_declare(queue='phya')
            channel.basic_consume(queue='phya', on_message_callback=process_message, auto_ack=True)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            logging.info("Failed to connect to RabbitMQ")
            time.sleep(10)


if __name__ == '__main__':
    main()
