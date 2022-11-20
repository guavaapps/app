import base64
import json
import logging

import numpy
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import pymongo
from pymongo import server_api
import bson

layers = keras.layers

VERSION = "dlstmp-1.0.0"

FEATURES = 5

GET = ["get", "GET"]
CREATE = ["create", "CREATE"]


class LSTMPCell(layers.LSTMCell):
    def __init__(self, units):
        super(LSTMPCell, self).__init__(units)

        self.units = units
        self.proj_w = tf.Variable(tf.random.normal([units, units]), trainable=True, name="proj_w")

    def call(self, inputs, states, training=None):
        h, c = states

        o, [_, c] = super(LSTMPCell, self).call(inputs, states, training)

        proj_h = tf.matmul(h, self.proj_w)

        return o, [proj_h, c]

    def get_config(self):
        config = super(LSTMPCell, self).get_config()
        config.update({
            "units": self.units,
            "proj_w": self.proj_w.numpy()
        })
        return config


class LSTMP(layers.RNN):
    def __init__(self, units, return_sequences=False):
        self.cell = LSTMPCell(units)
        super(LSTMP, self).__init__(cell=self.cell, return_sequences=return_sequences, stateful=False)

    def get_config(self):
        base_config = super(LSTMP, self).get_config()
        base_config.update({
            # "cell": self.cell,
            # "units": self.units,
            # "return_sequences": self.return_sequences
        })

        return base_config


def create_batches(dataset, look_back=1):
    x, y = [], []

    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]

        x.append(a)
        y.append(dataset[i + look_back])

    return np.array(x), np.array(y)


def scale_min_max(dataset):
    print(dataset.shape)

    temp = dataset.reshape(FEATURES, len(dataset))
    scaled = []

    for feature in temp:
        min = np.array(feature).min()
        max = np.array(feature).max()

        delta = max - min

        for timestep in feature:
            scaled.append((timestep - min) / delta)

    scaled = np.array(scaled).reshape(len(dataset), FEATURES)

    return scaled


def get_mongo_client():
    client = pymongo.MongoClient(
        "mongodb+srv://spotlight-dlstmp:spotlight-dlstmp@spotlight.zbz0fuh.mongodb.net/?retryWrites=true&w=majority",
        server_api=server_api.ServerApi('1'))

    return client


def get_model_config(client, user_id):
    print(client.list_databases())
    spotlight = client.Spotlight
    models = spotlight.Model
    model = models.find_one({"spotify_id": user_id})

    return model


def get_timeline(client, user_id):
    spotlight = client.Spotlight
    users = spotlight.User
    user = users.find_one({"spotify_id": user_id})

    if user is None:
        return None

    tracks = user["timeline"]

    timeline = [track.features for track in tracks]

    return timeline


def config_model(model, model_config):
    model_params = model_config["model_params"]

    params = [model_param["params"] for model_param in model_params]
    shapes = [model_param["shape"] for model_param in model_params]

    weights = [np.array(params[i]).reshape(shapes[i]) for i in range(len(params))]

    model.set_weights(weights)


def create_config(model, user_id):
    params = model.get_weights()

    p = []
    shapes = []
    dtypes = []

    for param in params:
        p.append(param.flatten().tolist())
        shapes.append(param.shape)
        dtypes.append(param.dtype)

    realm_params = [{
        "params": param.flatten().tolist(),
        "shape": param.shape
    } for param in params]

    config = {
        "_id": user_id,
        "model_params": realm_params
    }

    return config


def update_config(client, config):
    spotlight = client.Spotlight
    models = spotlight.Model

    models.replace_one({"_id": config["_id"]}, config, True)


def log_invoke(user_id, action, look_back, epochs):
    print(f"function invoked at - {int(time.time_ns() / 1000)}")
    print(f"    user_id={user_id}")
    print(f"    action={action}")
    print(f"training params - look_back={look_back} epochs={epochs}")


# event:
#   user_id - Spotify ID of the model
#   action - GET or CREATE
#   look_back - number of timesteps to look_back on when training the model
#   epochs
def lambda_handler(event, context):
    # cant stop thinking about you

    encoded = event["body"]
    b = base64.b64decode(encoded)
    body = json.loads(b)
    user_id = body["user_id"]
    look_back = body["look_back"]
    epochs = body["epochs"]
    action = body["action"]

    log_invoke(user_id, action, look_back, epochs)

    client = get_mongo_client()

    timeline = get_timeline(client, user_id)
    if timeline is None:
        print(f"Timeline is empty - model optimisation failed, returning unoptimised model (not recommended)")
        timeline = tf.random.normal([20, FEATURES]).numpy().astype("float32").tolist()

    timeline = scale_min_max(np.array(timeline))
    x, y = create_batches(timeline, look_back)

    input = layers.Input(shape=(look_back, FEATURES))  # 2 ts 4 f // 1, 4 ////// ts, f
    lstmp1 = LSTMP(64, return_sequences=True)(input)
    lstmp2 = LSTMP(64)(lstmp1)
    output = layers.Dense(FEATURES)(lstmp2)

    model = keras.Model(inputs=input, outputs=output, name="dlstmp")

    config = get_model_config(client, user_id)

    # if not (config is None):
    # config_model(model, config)

    model.compile(optimizer="adam", loss="mse")

    if action == "CREATE" and timeline is not None:
        model.fit(x, y, epochs=epochs)

        config = create_config(model, user_id)
        update_config(client, config)

    if config is None:
        config = create_config(model, user_id)
        update_config(client, config)

    model.save("/tmp/model")

    converter = tf.lite.TFLiteConverter.from_saved_model("/tmp/model")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()

    m = base64.b64encode(tflite_model)

    body = {
        "model": m.decode("utf-8"),
        "timestamp": int(time.time_ns() / 1000),
        "version": VERSION
    }

    optimised = True

    if timeline is None:
        optimised = False

    response = {
        "statusCode": 200,
        "optimised": optimised,
        "body": body
    }

    return json.dumps(response)
