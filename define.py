import os

OPENPACK_VERSION = "v0.3.1"
TRAINING_VERSION = "v5"
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

SAVE_CKPT_PATH = "/Users/farina/Workspace/Databases/OpenPack/ckpt/" + TRAINING_VERSION
if not os.path.exists(SAVE_CKPT_PATH):
    os.makedirs(SAVE_CKPT_PATH)

USER_ID_TRAIN = [
    "U0101",
    # "U0102",
    "U0103",
    # "U0105",
    # "U0106",
    # "U0107", "U0109", "U0111", "U0202", "U0205", "U0210"
]
SESSION_ID_TRAIN = [
    "S0100",
    # "S0300",
    # "S0200",
    # "S0400", "S0500"
]
DEVICE_ID = [["atr01", "atr02"]]
E4_DEVICE_ID = [['e401', 'e402']]

# ERROR_FILES = [["U0103", "S0100", "atr01"], ["U0103", "S0100", "atr02"],
#                ["U0107", "S0300", "atr01"], ["U0107", "S0300", "atr02"],
#                ["U0108", "S0400",  "atr01"], ["U0108", "S0400",  "atr02"],
#                ["U0109", "S0100", "atr01"], ["U0109", "S0100", "atr02"],
#                ["U0204", "S0100", "atr02"], ["U0204", "S0500", "atr02"],
#                ["U0207", "S0100", "atr02"]]

ERROR_FILES = [["U0103", "S0100"],
               ["U0107", "S0300"],
               ["U0108", "S0400"],
               ["U0109", "S0100"],
               ["U0204", "S0100"], ["U0204", "S0500"],
               ["U0207", "S0100"]]

ONE_SECOND_IN_MILISECOND = 1000  # ms

FS_ORG = 1000/33  # Hz
FS_TARGET = 30  # Hz

# WINDOW_SIZE = 60  # second
WINDOW_SIZE = 4  # second
# OVERLAP = 30  # second
OVERLAP = 2  # second

# NUM_DATA_TYPE = 11  # include unix_time column
NUM_DATA_TYPE = 10
NUM_CLASSES = 11
DATA_LEN = 30
# FEATURE_LEN = 7
FEATURE_LEN = 13
BATCH_SIZE = 8

LEARNING_RATE = 1e-3
EPOCHS = 3

FS_KEYPOINT = 15
FS_E4 = 4
FS_BVP = 64

# VISUALIZE
USER_ID_TRAIN_PLOT = [
    "U0101",
    # "U0102",
    # "U0103",
    # "U0105",
    # "U0106",
    # "U0107", "U0109", "U0111", "U0202", "U0205", "U0210"
]

SESSION_ID_TRAIN_PLOT = [
    "S0100",
    # "S0300",
    # "S0200",
    # "S0400", "S0500"
]

SAVE_IMG_PATH = DATASET_ROOTDIR + "/openpack/" + OPENPACK_VERSION + "/img"
if not os.path.exists(SAVE_IMG_PATH):
    os.makedirs(SAVE_IMG_PATH)

KP_POSITIONS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']