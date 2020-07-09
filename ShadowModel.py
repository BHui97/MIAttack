from dataLoader import *
from tensorflow.keras.applications import ResNet50, ResNet101, VGG16, VGG19, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation,Conv2D, MaxPooling2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import metrics
import configparser
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
config = configparser.ConfigParser()
config.read('shadow_model_config.ini')
# DATA_NAME = sys.argv[1]
# MODEL = sys.argv[2]
DATA_NAME = 'Diff_CH_MNIST'
MODEL = 'ResNet'
WEIGHTS = None if config['{}_{}'.format(DATA_NAME, MODEL)]['WEIGHTS'] == 'None' else \
    config['{}_{}'.format(DATA_NAME, MODEL)]['WEIGHTS']
EPOCHS = int(config['{}_{}'.format(DATA_NAME, MODEL)]['EPOCHS'])
SAVED_FOLADER = config['{}_{}'.format(DATA_NAME, MODEL)]['SAVED_FOLDER']
BATCH_SIZE = 64
LEARNING_RATE = float(config['{}_{}'.format(DATA_NAME, MODEL)]['LEARNING_RATE'])
WEIGHTS_PATH = "weights/{}/{}_{}.hdf5".format(SAVED_FOLADER, DATA_NAME, MODEL)
(x_train, y_train), (x_test, y_test), _ = globals()['load_' + DATA_NAME]('ShadowModel')


def create_ResNet101_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        ResNet101(include_top=False,
                 weights=WEIGHTS,
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_VGG16_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        VGG16(include_top=False,
                 weights=WEIGHTS,
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_VGG19_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        VGG19(include_top=False,
                 weights=WEIGHTS,
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_DenseNet121_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        DenseNet121(include_top=False,
                 weights=WEIGHTS,
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_CNN_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def create_Location_1_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Dense(128, input_shape=input_shape, activation='relu'),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_Location_2_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Dense(256, input_shape=input_shape, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_Location_3_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Dense(512, input_shape=input_shape, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_Location_5_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Dense(2048, input_shape=input_shape, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_Location_6_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Dense(4096, input_shape=input_shape, activation='relu'),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_ResNet_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        ResNet50(include_top=False,
                 weights=WEIGHTS,
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_simple_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def train(model, x_train, y_train, x_test, y_test):
    """
    Train the target model and save the weight of the model
    :param model: the model that will be trained
    :param x_train: the image as numpy format
    :param y_train: the label for x_train
    :param weights_path: path to save the model file
    :return: None
    """
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[checkpoint])


def evaluate(x_test, y_test):
    model = keras.models.load_model(WEIGHTS_PATH)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=5e-5),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
    F1_Score = 2 * (precision * recall) / (precision + recall)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (loss, accuracy, precision, recall, F1_Score))


TargetModel = globals()['create_{}_model'.format(MODEL)](x_train.shape[1:])

train(TargetModel, x_train, y_train, x_test, y_test)

evaluate(x_train, y_train)
evaluate(x_test, y_test)