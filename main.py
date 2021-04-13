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
from math import atan2, degrees

RABBITMQ_USERNAME = os.environ.get("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD")
RABBITMQ_QUEUE = os.environ.get("RABBITMQ_QUEUE")
PROD = os.environ.get("PROD") == "true"
API_TOKEN = os.environ.get("API_TOKEN")

def read_results(id_):
    with open("outputs/alpha_pose_{id}/alphapose-results.json".format(id=id_), 'r') as f:
        alpha_result = json.load(f)
    return alpha_result

def process_message(ch, model, properties, body):
    message = json.loads(body)
    job_id = message['job_id']
    try:
        inference_video('uploads/' + job_id + '.mp4', 'alpha_pose')
        with open("outputs/alpha_pose_{id}/alphapose-results.json".format(id=job_id), mode="r") as f:
            requests.put('http://api:8000/jobs/' + job_id,
                         data=({"status": "PROCESSED"}),
                         files={'file': f},
                         headers={"x-token": "test"}
                         )
    except Exception as e:
        requests.put('http://api:8000/jobs/' + job_id,
                     data=({"status": "FAILED"}),
                     headers = {"x-token": API_TOKEN}
        )
        raise e

def process_test_message(ch, model, properties, body):
    print("process_test_message")
    message = json.loads(body)
    job_id = message['job_id']
    with open("alphapose-results.json") as f:
        response = requests.put('http://api:8000/jobs/' + job_id,
                    data=({"status": "PROCESSED"}),
                    files={'file': f},
                    headers={"x-token": API_TOKEN}
                     )
        print(response)
        print(response.text)

def main():
    print("Starting running main")
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
            channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=process_message if PROD else process_test_message, auto_ack=True)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            logging.info("Failed to connect to RabbitMQ")
            time.sleep(10)


if __name__ == '__main__':
    if PROD:
        build()
    main()
