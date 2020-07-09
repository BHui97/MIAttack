from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from DiffMemUtil import evaluate_attack

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

DATANAME = 'CH_MNIST'
TARGET_MODEL_GENRE = 'ResNet'
SHADOW_MODEL_GENRE = 'ResNet'
NN_ATTACK_WEIGHTS_PATH = "weights/NN_Attack/DIFF_NN_Attack_{}_{}.hdf5".format(DATANAME, SHADOW_MODEL_GENRE)
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATANAME, TARGET_MODEL_GENRE)
SHADOW_WEIGHTS_PATH = "weights/DistributionShadow/Diff_{}_{}.hdf5".format(DATANAME, TARGET_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['load_' + DATANAME]('TargetModel')
Target_Model = load_model(TARGET_WEIGHTS_PATH)
def loss_threshold_attack(x_, y_true):
    (x_train_sha, y_train_sha), _, m_train = globals()['load_Diff_' + DATANAME]('ShadowModel')
    Shadow_Model = load_model(SHADOW_WEIGHTS_PATH)
    avg_loss = Shadow_Model.evaluate(x_train_sha, y_train_sha)[0]

    x_loss = np.asarray([-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in Target_Model.
                        predict(x_)[y_true.astype(bool)]])
    m_pred = np.where(x_loss <= avg_loss, 1, 0)
    return m_pred

m_pred = loss_threshold_attack(np.r_[x_train_tar, x_test_tar], np.r_[y_train_tar, y_test_tar])
evaluate_attack(m_pred, m_true)
