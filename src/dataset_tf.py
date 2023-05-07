from pathlib import Path
# import torch
import pandas as pd
import numpy as np
import tensorflow as tf
from const import DATA_PATH

# from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
# from transforms import get_transform
from itertools import islice
from typing import Tuple, Dict
from argparse import Namespace
# from torch.utils.data import DataLoader


def split_dataset(df, root_dir, isTraining, gender='a'):
    """
    Split dataframe into train/test, where test is `test_fold` and train is remaining folds
    Args:
        df (DataFrame): pandas DataFrame with annotations.
        nfolds (int): number of folds
        test_fold (int): test fold [1...nfolds]
        root_dir (string or Path): directory with radiographs
        gender ('m'|'f'|'a'): filter dataset based on gender, [m]ale, [f]emale, gender [a]gnostic
    """
    assert gender in ['a', 'm', 'f']
    if isTraining:
        image_dir = Path('data/train/boneage-training-dataset')
    else:
        image_dir = Path('data/test/boneage-test-dataset')
    root_dir = Path(root_dir)
    # image_dir = Path('data/hope/seeifworks')
    # root_dir = Path('data/hope')

    print(image_dir)
    print(root_dir)

    # make sure all listed radiographs are actually present
    radiograph_list = [int(f.stem)for f in image_dir.glob('*.png')]
    df = df.loc[df['id'].isin(radiograph_list)]
    print(len(radiograph_list))
    labels = df['boneage'].values.tolist()
    print(len(labels))
    
    if isTraining:
        return tf.keras.utils.image_dataset_from_directory(
            root_dir,
            labels=labels,
            label_mode='int',
            class_names=None,
            color_mode='grayscale',
            batch_size=2,
            image_size=(800, 600),
            shuffle=True,
            seed=17,
            validation_split=0.2,
            subset="both",
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False,
        )
    else:
        return tf.keras.utils.image_dataset_from_directory(
            root_dir,
            labels=labels,
            label_mode='int',
            class_names=None,
            color_mode='grayscale',
            batch_size=2,
            image_size=(800, 600),
            shuffle=True,
            seed=17,
            interpolation='bilinear',
        )
    # print(radiograph_list)

    if gender == 'm':
        df = df.loc[df['male']]
    elif gender == 'f':
        df = df.loc[~df['male']]
    df_len = len(df)
    # print(df_len)

    fold_length = int(np.ceil(df_len / nfolds))
    test_index = range(fold_length * (test_fold - 1), min(df_len, fold_length * test_fold))
    train_index = [i for i in range(df_len) if i not in test_index]
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    assert pd.concat([train_df, test_df]).loc[df.index].equals(df)

    return train_df, test_df



def get_loaders(args: Namespace):#-> Tuple[DataLoader, DataLoader]:
    train_annotation_frame = pd.read_csv(args.train_annotation_csv)
    train_df, validation_df = split_dataset(train_annotation_frame, args.train_data_dir, True, gender='a')
    return train_df, validation_df #data_frames

def get_test_loader(args: Namespace):
    test_annotation_frame = pd.read_csv(args.test_annotation_csv)
    test_df = split_dataset(test_annotation_frame, args.test_data_dir, False, gender='a')
    print(test_df)

    test_labels = np.concatenate([y for x, y in test_df], axis=0)
    test_images = np.concatenate([x for x, y in test_df], axis=0)


    # test_df = test_df.unbatch()
    # print(test_df)
    # test_images = [test_df.map(lambda x, y: x)]
    # test_labels = [test_df.map(lambda x, y: y)]
    return test_images, test_labels



# if __name__ == '__main__':
#     model_type = 'age'
#     bone_age_frame = pd.read_csv(DATA_PATH / 'train.csv')
#     root_dir = DATA_PATH / 'train'

#     nfolds = 9
#     # for test_fold in range(1, nfolds + 1):
#     #     train_df, test_df = split_dataset(bone_age_frame, test_fold, nfolds, 'data/train', gender='a')
#     #     print(len(train_df), len(test_df))
#     train_data,validation_df = split_dataset(bone_age_frame,4,6, 'data/train', gender='a')
#     print(train_data)