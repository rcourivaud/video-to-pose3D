import pika
import sys
import json
import logging
import time
import requests
import os
import numpy as np
from videopose import inference_video
from build import build
import math

RABBITMQ_USERNAME = os.environ.get("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD")
RABBITMQ_QUEUE = os.environ.get("RABBITMQ_QUEUE")

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


def get_2d_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    myradians = math.atan2(y2 - y1, x2 - x1)
    mydegrees = math.degrees(myradians)
    return mydegrees


def get_2d_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    distance = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return distance


def get_2d_velocity(p1, p2, fps=25):
    distance = get_2d_distance(p1, p2)
    velocity = distance * 1 / fps
    return velocity


def get_2d_acceleration(p1, p2, p3, fps=25):
    velo1 = get_2d_velocity(p1, p2, fps=fps)
    velo2 = get_2d_velocity(p2, p3, fps=fps)
    acceleration = (velo2 - velo1) * 1 / 25
    return acceleration


def _build_metas(v_0, v_1, v_2, i, fps=25):
    d = {}
    if i > 0:
        d["distance"] = get_2d_distance(v_0, v_1)
    else:
        d["distance"] = None
    if i > 0:
        d["velocity"] = get_2d_velocity(v_0, v_1, fps=fps)
    else:
        d["velocity"] = None

    if i > 1:
        d["acceleration"] = get_2d_acceleration(v_0,
                                                v_1,
                                                v_2, fps=fps)
    else:
        d["acceleration"] = None

    return d

def build_estimated_metadata(poses_2d, fps=25):
    data = []
    for i in range(0, len(poses_2d), FPS):
        pose_data = {}
        for joint in KEYPOINT_MAPPING.keys():
            x_0_slice = slice(i, i + fps)
            x_1_slice = slice(i - fps, i)
            x_2_slice = slice(i - 2 * fps, i - fps)
            x_0, y_0 = [e[joint]["x"] for e in poses_2d[x_0_slice]], [e[joint]["y"] for e in poses_2d[x_0_slice]]
            x_1, y_1 = [e[joint]["x"] for e in poses_2d[x_1_slice]], [e[joint]["y"] for e in poses_2d[x_1_slice]]
            x_2, y_2 = [e[joint]["x"] for e in poses_2d[x_2_slice]], [e[joint]["y"] for e in poses_2d[x_2_slice]]

            v_0 = np.array([np.mean(x_0), np.mean(y_0)])
            v_1 = np.array([np.mean(x_1), np.mean(y_1)])
            v_2 = np.array([np.mean(x_2), np.mean(y_2)])

            pose_data[joint] = _build_metas(v_0, v_1, v_2, i=i, fps=1)
        data.append(pose_data)
    return data


def build_metadata(poses_2d):
    data = []
    for i in range(len(poses_2d)):
        pose_data = {}
        for joint in KEYPOINT_MAPPING.keys():
            d = {}
            v_0 = np.array([poses_2d[i][joint]["x"], poses_2d[i][joint]["y"]])
            v_1 = np.array([poses_2d[i - 1][joint]["x"], poses_2d[i - 1][joint]["y"]]) if i > 0 else None
            v_2 = np.array([poses_2d[i - 2][joint]["x"], poses_2d[i - 2][joint]["y"]]) if i > 1 else None
            pose_data[joint] = _build_metas(v_0, v_1, v_2, i=i, fps=fps)
        data.append(pose_data)
    return data


def process_json_file(json_data):
    def _process_frame(keypoints, d2=False):
        xs = [elt for e, elt in enumerate(keypoints) if e % 3 == 0]
        ys = [elt for e, elt in enumerate(keypoints) if e % 3 == 1]
        zs = [elt for e, elt in enumerate(keypoints) if e % 3 == 2]
        keypoints_mapped = {
            name: {
                "x": xs[keypoint_index],
                "y": ys[keypoint_index],
            } if d2 else {
                "x": xs[keypoint_index],
                "y": ys[keypoint_index],
                "z": zs[keypoint_index] * 100,
            } for name, keypoint_index in KEYPOINT_MAPPING.items()
        }
        return keypoints_mapped

    d3_poses = [_process_frame(frame_result["keypoints"]) for frame_result in json_data]
    d2_poses = [_process_frame(frame_result["keypoints"], d2=True) for frame_result in json_data]
    metadata = build_metadata(d2_poses)
    estimated_metadata = build_estimated_metadata(d2_poses)
    processed_results = dict(
        poses=d3_poses,
        poses_2d=d2_poses,
        frames=len(json_data),
        metadata=metadata,
        estimated_metadata=estimated_metadata,
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
        requests.put('http://api:8000/jobs/' + job_id, json={'status': 'PROCESSED'})
    except Exception as e:
        requests.put('http://api:8000/jobs/' + job_id, json={'status': 'FAILED'})
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
            channel.queue_declare(queue=RABBITMQ_QUEUE)
            channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=process_message, auto_ack=True)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            logging.info("Failed to connect to RabbitMQ")
            time.sleep(10)


if __name__ == '__main__':
    build()
    main()
