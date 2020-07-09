import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import os
from PIL import Image
from tqdm import tqdm


def load_CH_MNIST(model_mode):
    """
    Loads CH_MNist dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `label_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    # Initialize Data
    dataframe = pd.read_csv('/home/bo/Project/DataLoader/data/CH_MNist/hmnist_64_64_L.csv')

    trainDF, testDF = train_test_split(dataframe, train_size=0.5,
                                       random_state=1 if model_mode == 'TargetModel' else 3,
                                       stratify=dataframe['label'].values)

    x_train = trainDF.iloc[:, range(4096)].values.reshape((-1, 64, 64, 1))
    y_train = tf.keras.utils.to_categorical([i - 1 for i in trainDF.loc[:, 'label']])
    m_train = np.ones(y_train.shape[0])

    x_test = testDF.iloc[:, range(4096)].values.reshape((-1, 64, 64, 1))
    y_test = tf.keras.utils.to_categorical([i - 1 for i in testDF.loc[:, 'label']])
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_CIFAR(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_CIFAR10(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_CUB(model_mode):
    """
    Loads CALTECH_BIRDS2011 (CUB_200) dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    x_train, y_train = tfds.load('caltech_birds2011', split='train' if model_mode == 'TargetModel' else 'test',
                                 batch_size=-1, as_supervised=True)

    x_test, y_test = tfds.load('caltech_birds2011', split='test' if model_mode == 'TargetModel' else 'train',
                               batch_size=-1, as_supervised=True)

    x_train = tf.image.resize(x_train, (200, 200))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=200)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.image.resize(x_test, (200, 200))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=200)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_EYE_PACS(model_mode):
    """
        Loads EyePACs dataset and maps it to Target Model and Shadow Model.
        :param model_mode: one of "TargetModel" and "ShadowModel".
        :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
        :raise: ValueError: in case of invalid `model_mode`.
        """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    mode = "target" if model_mode == "TargetModel" else "shadow"

    img_folder = "data/Eye_PACs/{}_images/".format(mode)

    label_df = pd.read_csv("data/Eye_PACs/{}_label.csv".format(mode), index_col=0)

    def set_data(img_path, label, desired_size=150):
        N = len(os.listdir(img_path))
        x_ = np.empty((N, 150, 150, 3), dtype=np.uint8)
        y_ = np.empty(N)
        for i, img_name in enumerate(tqdm(os.listdir(img_path))):
            x_[i, :, :, :] = Image.open(img_path + img_name).resize((desired_size,) * 2, resample=Image.LANCZOS)
            y_[i] = label.loc[img_name, 'level']
        y_ = tf.keras.utils.to_categorical(y_, num_classes=5)

        return x_, y_

    x_, y_ = set_data(img_folder, label_df)
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, train_size=0.5,
                                                        random_state=1 if model_mode == "TargetModel" else 3,
                                                        stratify=y_)
    m_train = np.ones(y_train.shape[0])
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_Location(model_mode):
    """
    Loads Location dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    y_data, x_data = np.load('data/location.npz').values()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.5,
                                                        random_state=1 if model_mode == 'TargetModel' else 3,
                                                        stratify=y_data)
    y_train = tf.keras.utils.to_categorical(y_train-1,num_classes=30)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test-1,num_classes=30)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_Diff_CH_MNIST(model_mode):
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    # Initialize Data
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.5,
                                                        random_state=1 if model_mode == 'TargetModel' else 3,
                                                        stratify=labels.numpy())

    y_train = tf.keras.utils.to_categorical(y_train-1, num_classes=8)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test-1, num_classes=8)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_Diff_CUB(model_mode):
    """
    Loads CALTECH_BIRDS2011 (CUB_200) dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    x_train, y_train, x_test, y_test = tfds.load('caltech_birds2010',
                                                 split=['train', 'test'] if model_mode == 'TargetModel' else 'test',
                                                 batch_size=-1, as_supervised=True)
    x_train, y_train, x_test, y_test = tfds.load('caltech_birds2011',
                                                 split=['train', 'test'] if model_mode == 'TargetModel' else 'test',
                                                 batch_size=-1, as_supervised=True)

    x_train = tf.image.resize(x_train, (200, 200))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=200)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.image.resize(x_test, (200, 200))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=200)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_EYE_PACs(model_mode):
    """
    Loads EyePACs dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    mode = "second" if model_mode=="TargetModel" else "third"

    base_path = "/home/bo/Project/Eyes_data"
    train_folder = "{}/{}_train/".format(base_path, mode)
    test_folder = "{}/{}_test/".format(base_path, mode)

    label_df = pd.read_csv("/home/bo/Project/Eyes_data/label.csv", error_bad_lines=False, index_col=0)
    def set_data(img_path, label, desired_size=150):
        N = len(os.listdir(img_path))
        x_ = np.empty((N, 150, 150, 3), dtype=np.uint8)
        y_ = np.empty(N)
        for i, img_name in enumerate(tqdm(os.listdir(img_path))):
            x_[i, :, :, :] = Image.open(img_path + img_name).resize((desired_size,) * 2, resample=Image.LANCZOS)
            y_[i] = label.loc[os.path.splitext(img_name)[0], 'level']
        y_ = tf.keras.utils.to_categorical(y_, num_classes=5)

        return x_, y_

    x_train, y_train = set_data(train_folder, label_df)
    m_train = np.ones(y_train.shape[0])
    x_test, y_test = set_data(test_folder, label_df)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member
