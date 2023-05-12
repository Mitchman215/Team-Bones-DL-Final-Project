from pathlib import Path
from typing import Dict, Any

import tensorflow as tf


class m46_tf(tf.keras.Model):
    def __init__(self, input_shape, model_type='age'):
        super().__init__()

        self.input_shapes = input_shape
        self.nchannel = input_shape[0]
        assert model_type in ['age', 'gender']
        self.model_type = model_type

        # Saving args for convinient restoring from ckpt
        self._params = {
            'input_shape': input_shape,
            'model_type': model_type
        }

        self.convolution = tf.keras.Sequential([
            m46_tf._vgg_block(32),  # Block 1
            m46_tf._vgg_block(64),  # Block 2
            m46_tf._vgg_block(128),  # Block 3
            m46_tf._vgg_block(128),  # Block 4
            m46_tf._vgg_block(256),  # Block 5
            m46_tf._vgg_block(384),  # Block 6
            tf.keras.layers.Flatten(),
        ], "Convolutions")

        self.fcc = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2048),
            tf.keras.layers.ELU(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2048),
            tf.keras.layers.ELU(),
            tf.keras.layers.Dense(1),
        ], "fcc")
        if self.model_type == 'gender':
            self.fcc.add(tf.keras.activations.sigmoid())

        if self.model_type == 'age':
            self._loss_function = tf.keras.losses.MeanAbsoluteError()
        else:
            self._loss_function = tf.keras.losses.BinaryCrossentropy()
        # self._initialize_weights() #DO WE NEED THIS

    @staticmethod
    def _vgg_block(out_channels, kernel_size=3):
        vgg_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                # in_channels=in_channels,
                filters=out_channels,
                kernel_size=kernel_size,
                padding="valid",
                kernel_initializer=tf.keras.initializers.HeNormal()), # missing fan_out and nonlinearity args provided to nn.init.kaiming_normal_
            tf.keras.layers.ELU(),
            tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05), # args determined by default of pytorch's BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            tf.keras.layers.Conv2D(
                # in_channels=out_channels,
                filters=out_channels,
                kernel_size=1,
                kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.ELU(),
            tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05),
            tf.keras.layers.MaxPool2D(
                pool_size=(3, 3),
                strides=(2, 2))
        ])
        return vgg_block


    def call(self, x, labels=None):
        x = self.convolution(x)
        outputs = self.fcc(x)
        return outputs


    @property
    def init_params(self) -> Dict[str, Any]:
        return self._params

    @property
    def loss_function(self):
        return self._loss_function


if __name__ == '__main__':
    # Create the model

    input_shape = (None, 2000, 1500, 1)
    model = m46_tf(input_shape,model_type='age')
    model.build(input_shape)
    
    # Print the model
    model.summary()
