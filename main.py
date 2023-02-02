import base64
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

from app import app

layers = keras.layers


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


input = layers.Input(shape=(4, 2))  # 2 ts 4 f // 1, 4
lstmp1 = LSTMP(64, return_sequences=True)(input)
lstmp2 = LSTMP(64)(lstmp1)
output = layers.Dense(2)(lstmp2)

model = keras.Model(inputs=input, outputs=output, name="dlstmp2")

# print ("model built")
#
# model.save ("model")
# converter = tf.lite.TFLiteConverter.from_saved_model("model")
#
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
#     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
# ]
#
# tflite_model = converter.convert()
#
# m = base64.b64encode(tflite_model)
#
# response = {
#     "body": m
# }
#
# # with open("model.tflite", "wb") as m:
# #     m.write(tflite_model)

timesteps = 10
features = 5
look_back = 1
epochs = 2
timeline = tf.random.normal([timesteps, features]).numpy().astype("float32")  # .tolist()

print(f"timeline - {timeline} {type(timeline)}")

b = json.dumps({
    "user_id": "s37s05am9tq6uxbi8skoqwwh5::test", #test_spotify_id",
    "action": "GET",
    "look_back": look_back,
    "epochs": epochs,
    "timeline": timeline.tolist ()
})

test_event = {
    "body": base64.b64encode(b.encode())
}

# response = app.app.lambda_handler(test_event, None)

# params = app.app.get_model_config("s37s05am9tq6uxbi8skoqwwh5")

# print(params)

response = app.lambda_handler(test_event, None)

# print (response)

# FEATURES = 5
#
# input = layers.Input(shape=(look_back, FEATURES))  # 2 ts 4 f // 1, 4 ////// ts, f
# lstmp1 = LSTMP(64, return_sequences=True)(input)
# lstmp2 = LSTMP(64)(lstmp1)
# output = layers.Dense(FEATURES)(lstmp2)
#
# model = keras.Model(inputs=input, outputs=output, name="dlstmp")
#
# model.compile(optimizer="adam", loss="mse")
#
# c = app.create_config(model, "spotify_id")