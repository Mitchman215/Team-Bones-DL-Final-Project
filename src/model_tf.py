from pathlib import Path
from typing import Dict, Any

# import torch
import tensorflow as tf
from collections import OrderedDict
# import torch.nn as nn


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
            self._vgg_block(self.nchannel, 32, 1),  # Block 1
            self._vgg_block(32, 64, 2),  # Block 2
            self._vgg_block(64, 128, 3),  # Block 3
            self._vgg_block(128, 128, 4),  # Block 4
            self._vgg_block(128, 256, 5),  # Block 5
            self._vgg_block(256, 384, 6),  # Block 6
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

    def _vgg_block(self, in_channels, out_channels, block_num, kernel_size=3):
        b = f'block{block_num}_'
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

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, tf.Conv2d):
    #             tf.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (tf.BatchNorm2d, tf.GroupNorm)):
    #             tf.init.constant_(m.weight, 1)
    #             tf.init.constant_(m.bias, 0)

    def call(self, x, labels=None):
        x = self.convolution(x)
        outputs = self.fcc(x)
        return outputs

    # def save(self, path_to_save: Path) -> None:
    #     checkpoint = {
    #         'model_state_dict': self.state_dict(),
    #         'checkpoint_data': {'params': self._params}
    #     }
    #     # TODO find tf equivalent of this
    #     # tf.keras.Model.save()
    #     # torch.save(checkpoint, path_to_save)
    #     self.save(path_to_save)
    #     print(f'Model saved to {path_to_save}.')

    @property
    def init_params(self) -> Dict[str, Any]:
        return self._params

    @property
    def loss_function(self):# -> nn.modules.Module:
        return self._loss_function

    @classmethod
    def from_ckpt(cls, checkpoint: Path) -> 'm46_tf':
        # ckpt = torch.load(checkpoint, map_location='cpu') if type(checkpoint) == Path else checkpoint
        # model = cls(**ckpt['checkpoint_data']['params'])
        # model.load_state_dict(ckpt['model_state_dict'])
        # if type(checkpoint) == Path:
        #     print(f'Model was loaded from {checkpoint}.')
        # else:
        #     print(f'Model was loaded from dictionary.')
        # return model
        new_model = tf.keras.models.load_model(checkpoint)
        new_model.summary()
        return new_model



# def convert_checkpoint(checkpoint: Path or OrderedDict,
#                        params: Dict[str, Any]) -> Dict[str, Any]:
#     """make use of previous checkpoint format"""

#     ckpt = torch.load(checkpoint, map_location='cpu') if isinstance(checkpoint, Path) else checkpoint
#     updates = {'.predictions.': '.preds.', '.prediction_probs.': '.probs.'}
#     for k in list(ckpt):
#         upd = next((u for u in updates if u in k), None)
#         if upd:
#             ckpt[k.replace(upd, updates[upd])] = ckpt.pop(k)
#     checkpoint = {
#         'model_state_dict': ckpt,
#         'checkpoint_data': {'params': params}
#     }
#     return checkpoint


if __name__ == '__main__':
    # Number of GPUs available. Use 0 for CPU mode.
    # ngpu = 1

    # # Decide which device we want to run on
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # # Create the model
    # input_shape = (1, 500, 375)
    # model = m46_tf(input_shape, model_type='age').to(device)

    # # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #     model = nn.DataParallel(model, list(range(ngpu)))

    # Print the model

    input_shape = (None, 2000, 1500, 1)
    model = m46_tf(input_shape,model_type='age')
    model.build(input_shape)
    model.summary()
