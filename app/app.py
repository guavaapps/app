import base64
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import pymongo
from pymongo import server_api

layers = keras.layers

# current optimiser version
VERSION = "dlstmp-1.0.0"

SUCCESS = 0
ERROR = 1


# single lstm cell layer with adjacent projection layer
# processes a single timestep in a sequence
class LSTMPCell(layers.LSTMCell):
    def __init__(self, units):
        super(LSTMPCell, self).__init__(units)

        # number of cells in the layer
        self.units = units

        # weights between lstm nodes and projection nodes
        self.proj_w = tf.Variable(tf.random.normal([units, units]), trainable=True, name="proj_w")

    def call(self, inputs, states, training=None):
        # get the hidden states of the lstm layer
        h, _ = states

        # get the outputs and cell states
        o, [_, c] = super(LSTMPCell, self).call(inputs, states, training)

        # multiply hidden states by the projection layer weights
        # before the next call
        proj_h = tf.matmul(h, self.proj_w)

        return o, [proj_h, c]

    # update the config of this layer to include references to the added
    # projection layer
    def get_config(self):
        config = super(LSTMPCell, self).get_config()
        config.update({
            "units": self.units,
            "proj_w": self.proj_w.numpy()
        })
        return config


# processes the whole sequence
class LSTMP(layers.RNN):
    def __init__(self, units, return_sequences=False):
        self.cell = LSTMPCell(units)
        super(LSTMP, self).__init__(cell=self.cell, return_sequences=return_sequences, stateful=False)


# create training batches
def create_batches(dataset, look_back=1):
    # each batch (in x) contains the number of timesteps equal to look_back
    # y contains the next timestep for each batch i.e. the target output when the batch is
    # fed into the rnn
    x, y = [], []

    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]

        x.append(a)
        y.append(dataset[i + look_back])

    return np.array(x), np.array(y)


# connect to the mongodb cluster
def get_mongo_client():
    client = pymongo.MongoClient(
        "mongodb+srv://spotlight-dlstmp:spotlight-dlstmp@spotlight.zbz0fuh.mongodb.net/?retryWrites=true&w=majority",
        server_api=server_api.ServerApi('1'))

    return client


# get the model config document for this model from mongodb
def get_model_config(client, user_id):
    print(client.list_databases())
    spotlight = client.Spotlight
    models = spotlight.Model
    model = models.find_one({"_id": user_id})

    return model


# extract model weights from the current model config document and apply to model
def config_model(model, model_config):
    model_params = model_config["model_params"]

    # get flat list of weights
    params = [model_param["params"] for model_param in model_params]
    # get shapes of weight lists
    shapes = [model_param["shape"] for model_param in model_params]

    # reshape the flat lists into the required shapes
    weights = [np.array(params[i]).reshape(shapes[i]) for i in range(len(params))]

    model.set_weights(weights)


# create model config document from current model
def create_config(model, user_id):
    params = model.get_weights()

    # extract weight lists and their shapes
    # store weights as flat lists
    realm_params = [{
        "params": param.flatten().tolist(),
        "shape": param.shape
    } for param in params]

    # create document
    config = {
        "_id": user_id,
        "timestamp": int(time.time_ns() / 1000),
        "version": VERSION,
        "model_params": realm_params
    }

    return config


# update the remote model config document
def update_config(client, config):
    # cluster name
    spotlight = client.Spotlight

    # collection containing the model configs
    models = spotlight.Model

    models.replace_one({"_id": config["_id"]}, config, True)


def log_invoke(user_id, action, look_back, epochs):
    print(f"function invoked at - {int(time.time_ns() / 1000)}")
    print(f"    user_id={user_id}")
    print(f"    action={action}")
    print(f"training params - look_back={look_back} epochs={epochs}")


def lambda_handler(event, context):
    # get the base 64 encoded request body
    encoded = event["body"]
    # decode to string
    b = base64.b64decode(encoded)
    # to dict
    body = json.loads(b)

    # extract params
    # spotify id of the user requesting the model
    user_id = body["user_id"]
    # number of timesteps to train the model on
    look_back = body["look_back"]
    # number of iterations to use when training
    epochs = body["epochs"]
    # get model from existing config or create a new optimised model
    action = body["action"]
    # training dataset
    timeline = body["timeline"]

    client = get_mongo_client()

    print("mongo client obtained")

    config = get_model_config(client, user_id)

    if action == "CREATE":
        features = len(timeline[0])
    elif action == "GET" and config is not None:
        output_layer = config["model_params"][-1]
        features = output_layer["shape"][-1]
    else:
        # model cannot be created or obtained
        response = {
            # status code is 200 so that the client only has to listen to a
            # single stream
            "statusCode": 200,
            "status": ERROR,
            "body": {
                "model": None,
                "timestamp": None,
                "version": None,
            }
        }

        return json.dumps(response)

    print(f"input shape obtained [{features}]")

    # build model
    input = layers.Input(shape=(look_back, features))  # 2 ts 4 f // 1, 4 ////// ts, f
    # user 64 cells in each lstmp layer
    lstmp1 = LSTMP(64, return_sequences=True)(input)
    lstmp2 = LSTMP(64)(lstmp1)
    output = layers.Dense(features)(lstmp2)

    model = keras.Model(inputs=input, outputs=output, name="dlstmp")

    # use the adam optimiser and mean-squared error loss function
    model.compile(optimizer="adam", loss="mse")

    print("created model")

    # optimise model
    if action == "CREATE" and timeline is not None:
        x, y = create_batches(timeline, look_back)

        # train
        model.fit(x, y, epochs=epochs)

        # update the remote config
        config = create_config(model, user_id)
        update_config(client, config)

        print("model optimised")

    if config is None:
        config = create_config(model, user_id)
        update_config(client, config)

        print("config not found, creating new config")

    # update the model
    config_model(model, config)
    print("model configured")

    # temporarily save the model to a local filesystem
    model.save("/tmp/model")

    # create a tf lite model converter
    converter = tf.lite.TFLiteConverter.from_saved_model("/tmp/model")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    # convert the saved model to tf lite format
    tflite_model = converter.convert()

    # convert the tf lite model to a utf-8 string
    m = base64.b64encode(tflite_model)

    body = {
        "model": m.decode("utf-8"),
        # return timestamp for version tracking
        "timestamp": config["timestamp"],
        "version": VERSION
    }

    response = {
        "statusCode": 200,
        "status": SUCCESS,
        "body": body
    }

    return json.dumps(response)
