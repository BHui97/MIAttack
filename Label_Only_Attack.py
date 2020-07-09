from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from DiffMemUtil import evaluate_attack

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATANAME = 'CIFAR'
TARGET_MODEL_GENRE = 'ResNet'
NN_ATTACK_WEIGHTS_PATH = "weights/NN_Attack/NN_Attack_{}_{}.hdf5".format(DATANAME, TARGET_MODEL_GENRE)
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATANAME, TARGET_MODEL_GENRE)
SHADOW_WEIGHTS_PATH = "weights/Shadow/{}_{}.hdf5".format(DATANAME, TARGET_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['load_' + DATANAME]('TargetModel')

Target_Model = load_model(TARGET_WEIGHTS_PATH)

def Label_Only_Attack(x_, y_true):
    y_pred = Target_Model.predict_classes(x_)
    y_true = y_true.argmax(axis=1)
    m_pred = np.where(np.equal(y_pred, y_true), 1, 0)
    return m_pred

m_pred = Label_Only_Attack(np.r_[x_train_tar, x_test_tar], np.r_[y_train_tar, y_test_tar])
evaluate_attack(m_true, m_pred)
