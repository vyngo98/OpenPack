import os

OPENPACK_VERSION = "v0.3.0"
TRAINING_VERSION = "v2"
DATASET_ROOTDIR = "/Users/farina/Workspace/Databases/OpenPack/data/datasets"
TFRECORD_TRAIN_PATH = DATASET_ROOTDIR + "/" + TRAINING_VERSION + "/tfrecord_training"
if not os.path.exists(TFRECORD_TRAIN_PATH):
    os.makedirs(TFRECORD_TRAIN_PATH)

TFRECORD_TEST_PATH = DATASET_ROOTDIR + "/" + TRAINING_VERSION + "/tfrecord_testing"
if not os.path.exists(TFRECORD_TEST_PATH):
    os.makedirs(TFRECORD_TEST_PATH)

TFRECORD_VALID_PATH = DATASET_ROOTDIR + "/" + TRAINING_VERSION + "/tfrecord_valid"
if not os.path.exists(TFRECORD_VALID_PATH):
    os.makedirs(TFRECORD_VALID_PATH)

SAVE_CKPT_PATH = "/Users/farina/Workspace/Databases/OpenPack/ckpt/"
if not os.path.exists(SAVE_CKPT_PATH):
    os.makedirs(SAVE_CKPT_PATH)

USER_ID_TRAIN = [
    # "U0101",
    "U0102",
    # "U0103", "U0105", "U0106", "U0107", "U0109", "U0111", "U0202", "U0205", "U0210"
]
SESSION_ID_TRAIN = [
    "S0100",
    # "S0200",
    # "S0400", "S0500"
]
DEVICE_ID = ["atr01"]
E4_DEVICE_ID = ['e401']

ONE_SECOND_IN_MILISECOND = 1000  # ms

FS_ORG = 1000/33  # Hz
FS_TARGET = 30  # Hz

# WINDOW_SIZE = 60  # second
WINDOW_SIZE = 1  # second
# OVERLAP = 30  # second
OVERLAP = 0  # second

# NUM_DATA_TYPE = 11  # include unix_time column
NUM_DATA_TYPE = 10
NUM_CLASSES = 11
DATA_LEN = 30
FEATURE_LEN = 7
BATCH_SIZE = 8

LEARNING_RATE = 1e-3
EPOCHS = 3
