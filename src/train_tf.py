from argparse import Namespace
import warnings
from model_tf import m46_tf
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import numpy as np

from model import m46
from dataset_tf import get_loaders
from const import LOG_DIR, DATA_PATH, MODELS_DIR

def main(args):
    input_shape = (int(args.crop_size[0] * args.scale), int(args.crop_size[1] * args.scale), 1)
    print('Input shape', 'x'.join(map(str, input_shape)), '[HxWxC]')

    train_loader,validation_loader = get_loaders(args)
    
    model = m46_tf(input_shape=input_shape, model_type=args.model_type)
    optimizer = Adam(lr=2e-5, beta_1=0.5, beta_2=0.999)

    model.compile(optimizer=optimizer, loss=model.loss_function, metrics=['accuracy'])

    callbacks = [EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)]
    checkpoint_path = str(args.logdir / "model.ckpt")
    callbacks += [ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)]

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory=args.train_data_dir,
        target_size=(input_shape[0], input_shape[1]),
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode="binary",
        shuffle=True,
        seed=args.seed
    )

    validation_generator = test_datagen.flow_from_directory(
        directory=args.train_data_dir,
        target_size=(input_shape[0], input_shape[1]),
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode="binary",
        shuffle=True,
        seed=args.seed
    )

    history = model.fit(
        train_generator,
        epochs=args.n_epoch,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=True
    )
    model.save_weights(filepath=str(args.logdir / "model.h5"))

    # model.save(filepath=str(args.logdir / "model.h5"),save_format="tf")

if __name__ == '__main__':
    train_data_dir = DATA_PATH / 'hope'#'train'
    test_data_dir = DATA_PATH / 'test'
    crop_center = (1040, 800)
    crop_size = (2000, 1500)
    scale = 0.25
    train_annotation_csv = DATA_PATH / 'hoping.csv'#'train.csv'
    test_annotation_csv = DATA_PATH / 'test.csv'
    dataset_split = (10, 10)
    model_type = 'age'
    prev_ckpt = None
    model_save_dir = MODELS_DIR
    n_epoch = 5
    batch_size = 4
    n_gpu = 1
    n_workers = 0
    seed = 42
    device = '/GPU:0'
    logdir = LOG_DIR

    args = Namespace(
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        crop_center=crop_center,
        crop_size=crop_size,
        scale=scale,
        train_annotation_csv=train_annotation_csv,
        test_annotation_csv=test_annotation_csv,
        dataset_split=dataset_split,
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
