from dataLoader import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation,Conv2D, MaxPooling2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import metrics


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = 'CIFAR'
MODEL = 'ResNet'
WEIGHTS = 'imagenet'
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, MODEL)
(x_train, y_train), (x_test, y_test), _ = globals()['load_' + DATA_NAME]('TargetModel')


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


def create_LocationC_model(input_shape, num_classes=y_train.shape[1]):
    model = tf.keras.Sequential([
        Dense(1024, input_shape=input_shape, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def train(model, x_train, y_train):
    """
    Train the target model and save the weight of the model
    :param model: the model that will be trained
    :param x_train: the image as numpy format
    :param y_train: the label for x_train
    :param weights_path: path to save the model file
    :return: None
    """
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=5e-5),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train,
              y_train,
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

train(TargetModel, x_train, y_train)

evaluate(x_train, y_train)
evaluate(x_test, y_test)
