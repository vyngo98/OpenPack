# import os
# import tqdm
import numpy as np
import tensorflow as tf
from functools import partial
# from tensorflow.keras.utils import Progbar

from define import *


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _dtype_feature(ndarray):
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:
        raise ValueError("The input should be numpy ndarray. \
                           Instead got {}".format(ndarray.dtype))


def generate_data(path, dataset, classes=NUM_CLASSES):
    assert np.array(dataset[0]).shape[0] == np.array(dataset[1]).shape[0]
    if os.path.exists(path):
        os.remove(path)

    with tf.io.TFRecordWriter(path) as writer:
        label = tf.keras.utils.to_categorical(dataset[1], classes).astype('int64')
        for idx, data in enumerate(dataset[0]):
            feature = dict()
            feature['data'] = _float_feature(data.flatten().tolist())
            feature['feature'] = _float_feature(dataset[2][idx].tolist())
            feature['label'] = _int64_feature(label[idx].flatten().tolist())

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def _parse_function(record_batch, data_len, classes_len, feature_len):
    feature = {
        'data': tf.io.FixedLenFeature([data_len*NUM_DATA_TYPE, 1], tf.float32),
        # 'label': tf.io.FixedLenFeature([classes_len*WINDOW_SIZE], tf.int64),
        'label': tf.io.FixedLenFeature([classes_len], tf.int64),
        'feature': tf.io.FixedLenFeature([feature_len*NUM_DATA_TYPE, 1], tf.float32)
    }

    example = tf.io.parse_example(record_batch, feature)
    example['label'] = tf.cast(example['label'], tf.int8)
    return example['data'], example['label'], example['feature']


def _calc_num_steps(num_samples, batch_size):
    return (num_samples + batch_size - 1) // batch_size


def get_dataset_from_tfrecord(files):
    data, feature, label = [], [], []
    ds = tf.data.TFRecordDataset(files)
    map_function = partial(_parse_function, data_len=WINDOW_SIZE*FS_TARGET, classes_len=NUM_CLASSES, feature_len=FEATURE_LEN)
    ds = ds.map(map_function, num_parallel_calls=os.cpu_count())

    # _parse_function(ds, data_len=WINDOW_SIZE*FS_TARGET, classes_len=NUM_CLASSES, feature_len=FEATURE_LEN)
    ds = ds.batch(BATCH_SIZE)
    for index, batch in enumerate(ds):
        list_tensors = [i for i in batch]
        data.extend([tensor.numpy().flatten().reshape((WINDOW_SIZE*FS_TARGET, NUM_DATA_TYPE)) for tensor in list_tensors[0]])
        feature.extend([tensor.numpy().flatten() for tensor in list_tensors[2]])
        label.extend([tensor.numpy().flatten() for tensor in list_tensors[1]])
    return data, feature, label


# data, feature, label = get_dataset_from_tfrecord('/Users/farina/Workspace/Databases/OpenPack/data/datasets/v0.3.0/tfrecord_training/U0102-S0100_atr01_e401.tfrecord')
# a=0

# result = []
# for example in tf.compat.v1.python_io.tf_record_iterator("/Users/farina/Downloads/ann2/default.tfrecord"):
#     result.append(tf.train.Example.FromString(example))
#
# a=0