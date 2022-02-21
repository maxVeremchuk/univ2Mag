from flask import Flask, request, Response
from viberbot import Api
from viberbot.api.bot_configuration import BotConfiguration
from viberbot.api.messages import PictureMessage
from viberbot.api.messages.text_message import TextMessage


from viberbot.api.viber_requests import ViberConversationStartedRequest
from viberbot.api.viber_requests import ViberFailedRequest
from viberbot.api.viber_requests import ViberMessageRequest
from viberbot.api.viber_requests import ViberSubscribedRequest
from viberbot.api.viber_requests import ViberUnsubscribedRequest
from viberbot.api.event_type import EventType

import os
import time
import sched
import logging
import threading
import cv2
import urllib
import numpy as np
import requests

from filestack import Client, Filelink
client_filestack = Client(os.environ.get('FILELINK_TOKEN'))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__)
viber = Api(BotConfiguration(
    name='facerecognitionbot',
    avatar='https://www.kindpng.com/picc/m/381-3819158_facial-recognition-facial-image-recognition-logo-hd-png.png',
    auth_token=os.environ.get('AUTH_TOKEN')
))

@app.route('/', methods=['GET'])
def incoming_get():
    return Response(status=200)

@app.route('/', methods=['POST'])
def incoming():
    viber_request = viber.parse_request(request.get_data())
    if isinstance(viber_request, ViberMessageRequest):
        message = viber_request.message
        if isinstance(message, PictureMessage):
            logger.warning(message.media)
            filelink = client_filestack.upload_url(message.media)
            spitted_url = filelink.url.split('/')
            new_url = spitted_url[:-1] + ['detect_faces=export:true'] + [spitted_url[-1]]
            new_url = '/'.join(new_url)
            logger.warning(new_url)

            viber_img = urllib.request.urlopen(filelink.url)
            viber_img_cv = np.asarray(bytearray(viber_img.read()), dtype=np.uint8)
            viber_img_cv = cv2.imdecode(viber_img_cv, -1)

            response = requests.get(new_url)
            logger.warning(response.json())
            for json in response.json():
                viber_img_cv = cv2.rectangle(viber_img_cv, 
                                            (json['x'], json['y']), 
                                            (json['x']+json['width'], json['y']+json['height']), 
                                            (128,128,0), 2)

            cv2.imwrite('./message.jpg', viber_img_cv)

            filelink = client_filestack.upload(filepath="./message.jpg")
            message = PictureMessage(media=filelink.url)
            logger.warning(message)
            viber.send_messages(viber_request.sender.id, [
                message
            ])
        elif isinstance(message, TextMessage):
            viber.send_messages(viber_request.sender.id, [
                TextMessage(text="Please send photo!")
            ])
    elif isinstance(viber_request, ViberSubscribedRequest):
        viber.send_messages(viber_request.get_user.id, [
            TextMessage(text="thanks for subscribing!")
        ])

    return Response(status=200)

def set_webhook(viber, port):
	viber.set_webhook(f'https://frecbot.herokuapp.com/')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8443))

    scheduler = sched.scheduler(time.time, time.sleep)
    scheduler.enter(5, 1, set_webhook, (viber,port,))
    t = threading.Thread(target=scheduler.run)
    t.start()

    app.run(host='0.0.0.0', port=port, debug=True)
