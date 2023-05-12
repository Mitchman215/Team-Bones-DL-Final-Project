from argparse import Namespace
import warnings

from matplotlib import pyplot as plt
from model_tf import m46_tf
warnings.filterwarnings("ignore")

import tensorflow as tf
from pathlib import Path
import numpy as np

from dataset_tf import get_test_loader

from const import LOG_DIR, DATA_PATH, MODELS_DIR

def main(args):
    input_shape = (int(args.crop_size[0] * args.scale), int(args.crop_size[1] * args.scale), 1)
    print('Input shape', 'x'.join(map(str, input_shape)), '[HxWxC]')

    test_loader_images, test_labels = get_test_loader(args)

    model = m46_tf(input_shape=input_shape, model_type=args.model_type)
    optimizer = tf.keras.optimizers.Adam(lr=2e-5, beta_1=0.5, beta_2=0.999)

    model.compile(optimizer=optimizer, loss=model.loss_function, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    input_shape = (None, 800, 600, 1)
    model.build(input_shape)
    model.summary()

    

    checkpoint_path = str(args.logdir / "model.h5")
    # checkpoint_path = str(args.logdir / "model_weights.h5")
    # str(args.logdir / "model.ckpt/variables/variables.index")
    # str(args.logdir / "model.h5") 
    model.load_weights(filepath=checkpoint_path)

    loss, acc = model.evaluate(test_loader_images, test_labels, verbose=2)
    # y_pred = model.predict(test_loader_images, verbose=2)
    # print(y_pred)
    print("Restored model, accuracy: {:5.2f}".format(acc))

    # Plot of testing results
    # plt.scatter(test_labels, y_pred, label='Data')
    # p1 = max(max(y_pred), max(test_labels))
    # p2 = min(min(y_pred), min(test_labels))
    # plt.plot([p1, p2], [p1, p2], color='k', label='Predictions')
    # # plt.plot(x, y, color='k', label='Predictions')
    # plt.xlabel('True Bone Age')
    # plt.ylabel('Predicted Bone Age')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    test_data_dir = DATA_PATH / 'test'
    crop_center = (1040, 800)
    crop_size = (2000, 1500)
    scale = 0.25
    test_annotation_csv = DATA_PATH / 'test.csv'
    model_type = 'age'
    prev_ckpt = None
    model_save_dir = MODELS_DIR
    n_epoch = 3
    batch_size = 3
    n_gpu = 1
    n_workers = 0
    seed = 42
    device = '/GPU:0'
    logdir = LOG_DIR

    args = Namespace(
        test_data_dir=test_data_dir,
        crop_center=crop_center,
        crop_size=crop_size,
        scale=scale,
        test_annotation_csv=test_annotation_csv,
        model_type=model_type,
        prev_ckpt=prev_ckpt,
        model_save_dir=model_save_dir,
        n_epoch=n_epoch,
        batch_size=batch_size,
        n_gpu=n_gpu,
        n_workers=n_workers,
        seed=seed,
        device=device,
        logdir=logdir
    )

main(args)