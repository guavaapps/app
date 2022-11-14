# import base64
# import http.client
# import io
# import json
# import socket
#
import json
import resource

import numpy as np
import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import time
# import pymongo
# from pymongo import server_api
# import bson
# import dns
#
# layers = keras.layers

FEATURES = 5


# class LSTMPCell(layers.LSTMCell):
#     def __init__(self, units):
#         super(LSTMPCell, self).__init__(units)
#
#         self.units = units
#         self.proj_w = tf.Variable(tf.random.normal([units, units]), trainable=True, name="proj_w")
#
#     def call(self, inputs, states, training=None):
#         h, c = states
#
#         o, [_, c] = super(LSTMPCell, self).call(inputs, states, training)
#
#         proj_h = tf.matmul(h, self.proj_w)
#
#         return o, [proj_h, c]
#
#     def get_config(self):
#         config = super(LSTMPCell, self).get_config()
#         config.update({
#             "units": self.units,
#             "proj_w": self.proj_w.numpy()
#         })
#         return config
#
#
# class LSTMP(layers.RNN):
#     def __init__(self, units, return_sequences=False):
#         self.cell = LSTMPCell(units)
#         super(LSTMP, self).__init__(cell=self.cell, return_sequences=return_sequences, stateful=False)
#
#     def get_config(self):
#         base_config = super(LSTMP, self).get_config()
#         base_config.update({
#             # "cell": self.cell,
#             # "units": self.units,
#             # "return_sequences": self.return_sequences
#         })
#
#         return base_config
#
#
# def create_batches(dataset, look_back=1):
#     x, y = [], []
#
#     for i in range(len(dataset) - look_back - 1):
#         a = dataset[i:(i + look_back)]
#
#         x.append(a)
#         y.append(dataset[i + look_back])
#
#     return np.array(x), np.array(y)
#
#
# def scale_min_max(dataset):
#     print(dataset.shape)
#
#     temp = dataset.reshape(FEATURES, len(dataset))
#     scaled = []
#
#     for feature in temp:
#         min = np.array(feature).min()
#         max = np.array(feature).max()
#
#         delta = max - min
#
#         for timestep in feature:
#             scaled.append((timestep - min) / delta)
#
#     scaled = np.array(scaled).reshape(len(dataset), FEATURES)
#
#     return scaled
#
#
# def get_mongo_client():
#     client = pymongo.MongoClient(
#         "mongodb+srv://spotlight-dlstmp:spotlight-dlstmp@spotlight.zbz0fuh.mongodb.net/?retryWrites=true&w=majority",
#         server_api=server_api.ServerApi('1'))
#
#     return client
#
#
# def get_model_config(user_id):
#     client = get_mongo_client()
#
#     print(client.list_databases())
#     spotlight = client.Spotlight
#     models = spotlight.Model
#     model = models.find_one({"spotify_id": user_id})
#
#     if model is None:
#         model_params = {
#             "params": [],
#             "shapes": []
#         }
#
#         m = {
#             "_id": bson.ObjectId(),
#             "spotify_id": user_id,
#             "timeline": [[]],
#             "model_params": [model_params]
#         }
#
#         models.insert_one(m)
#
#     return model
#
#
# def get_timeline(user_id):
#     client = get_mongo_client()
#
#     spotlight = client.Spotlight
#     users = spotlight.User
#     user = users.find_one({"spotify_id": user_id})
#
#     if user is None:
#         return None
#
#     tracks = user["timeline"]
#
#     timeline = [track.features for track in tracks]
#
#     return timeline
#
#
# def config_model(model, model_config):
#     model_params = model_config["model_params"]
#
#     params = model_params["params"]
#     shapes = model_params["shapes"]
#
#     weights = [np.array(params[i]).reshape(shapes[i]) for i in range(len(params))]
#
#     model.set_weights(weights)
#
#
# def create_config(model, user_id):
#     params = model.get_weights()
#
#     bytes = []
#     shapes = []
#     dtypes = []
#
#     for param in params:
#         bytes.append(param.flatten())
#         shapes.append(param.shape)
#         dtypes.append(param.dtype)
#
#     model_params = {
#         "params": bytes,
#         "shapes": shapes
#     }
#
#     config = {
#         "_id": bson.ObjectId(),
#         "spotify_id": user_id,
#         "model_params": model_params
#     }
#
#     return config
#
#
# def update_config(config):
#     client = get_mongo_client()
#
#     spotlight = client.Spotlight
#     models = spotlight.Model
#
#     models.replace_one(config, True)
#
#
# def log_invoke(user_id, action, look_back, epochs):
#     print(f"function invoked at - {int(time.time_ns() / 1000)}")
#     print(f"    user_id={user_id}")
#     print(f"    action={action}")
#     print(f"training params - look_back={look_back} epochs={epochs}")
#
#
# def get_public_record():
#     client = get_mongo_client()
#     spotlight = client.Spotlight
#
#     users = spotlight.User.find()
#
#     features = [user.timeline.features for user in users]


# event:
#   user_id - Spotify ID of the model
#   action - GET or CREATE
#   look_back - number of timesteps to look_back on when training the model
#   epochs
def lambda_handler(event, context):
    print("hehe")
    print(event)
    print(context)

    user_id = event["body"]["user_id"]
    look_back = event["body"]["look_back"]
    epochs = event["body"]["epochs"]
    action = event["body"]["action"]

    print("body")

    # log_invoke(user_id, action, look_back, epochs)

    # timeline = get_timeline(user_id)
    # if timeline is None:
    #     print(f"Timeline is empty - model optimisation failed, returning unoptimised model (not recommended)")
    #
    # x, y = create_batches(timeline, look_back)
    #
    # input = layers.Input(shape=(look_back, FEATURES))  # 2 ts 4 f // 1, 4 ////// ts, f
    # lstmp1 = LSTMP(64, return_sequences=True)(input)
    # lstmp2 = LSTMP(64)(lstmp1)
    # output = layers.Dense(FEATURES)(lstmp2)
    #
    # model = keras.Model(inputs=input, outputs=output, name="dlstmp")
    #
    # config = get_model_config(user_id)
    #
    # if not (config is None):
    #     config_model(model, config)
    #
    # model.compile(optimizer="adam", loss="mse")
    #
    # if action == "CREATE":
    #     model.fit(x, y, epochs=epochs, batch_size=64)
    #
    #     params = model.get_weights()
    #
    #     config = create_config(model, user_id)
    #     update_config(config)
    #
    # bytes = []
    # shapes = []
    # dtypes = []
    #
    # d = []
    #
    # for param in params:
    #     bytes.append(param.tobytes())
    #     shapes.append(param.shape)
    #     dtypes.append(param.dtype)
    #
    #     if param.dtype not in d:
    #         d.append(param.dtype)
    #
    # new_p = []
    #
    # for i, b in enumerate(bytes):
    #     new_p.append(np.frombuffer(b, dtype=dtypes[i]).reshape(shapes[i]))
    #
    # for i in range(len(params)):
    #     p = params[i]
    #     n = new_p[i]
    #
    #     print(f"param {i}")
    #     print(f"   {np.array_equal(n, p)}")
    #
    # print(f"equal - {np.array_equal(new_p, params)}")
    #
    # print(f"dtypes - {d}")
    #
    # model.save("model")
    #
    # converter = tf.lite.TFLiteConverter.from_saved_model("model")
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    # ]
    #
    # # converter._experimental_lower_tensor_list_ops = True
    #
    # tflite_model = converter.convert()
    #
    # with open("model.tflite", "wb") as m:
    #     m.write(tflite_model)
    #
    # m = base64.b64encode(tflite_model)
    #
    # body = {
    #     "model": m,
    #     "timestamp": int(time.time_ns() / 1000),
    #     "version": "dlstmp-1.0.0"
    # }
    #
    # response = {
    #     "statusCode": 200,
    #     "body": body  # json.dumps(body)
    # }
    #
    # return response

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r /= 1000000

    print (f"{r} MB")

    return {
        "statusCode": 200
    }
