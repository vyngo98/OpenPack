import matplotlib.pyplot as plt

import define as df
from processing_data import *
import tfrecord as tfrecord
from model import ActivityModel
from callback import CreateCallBack

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def extract_feature(data):
    mean_ft = np.array(data.mean())[1:]
    std_ft = np.array(data.std())[1:]
    max_ft = np.array(data.max())[1:]
    min_ft = np.array(data.min())[1:]
    var_ft = np.array(data.var())[1:]
    features = np.array([mean_ft, std_ft, max_ft, min_ft, var_ft]).T.flatten()
    return features


def process_data(user_id, session_id, device_id, e4_device_id, openpack_version, dataset_rootdir, tfrecord_path,
                 feature=True, write_tfrecord=True):
    all_data = LoadData(user_id, session_id, device_id, e4_device_id, openpack_version, dataset_rootdir).process()
    annotation = all_data[0]
    data = all_data[1]

    # sort data based on timestamp
    annotation.sort_values('unixtime')
    data.sort_values('unixtime')

    # drop duplicate
    annotation = annotation.drop_duplicates()
    data = data.drop_duplicates()

    # interpolate missing values
    data_itpl = data.interpolate()
    data_itpl = data_itpl.fillna(0)

    annotation_itpl = annotation.interpolate()
    annotation_itpl = annotation_itpl.fillna(0)

    annotation_itpl["cls_idx"] = optk.OPENPACK_OPERATIONS.convert_id_to_index(annotation_itpl["operation"])

    # segmentation
    data_seg = []
    label_seg = []
    feature_seg = []
    # timestamp_ann = annotation_itpl['unixtime']
    # timestamp_data = data_itpl['unixtime']
    for i in range(len(annotation_itpl)):
        seg = data_itpl[(data_itpl["unixtime"] >= annotation_itpl['unixtime'][i]) & (data_itpl["unixtime"] <= annotation_itpl['unixtime'][i] + df.ONE_SECOND_IN_MILISECOND)]
        data_seg.append(np.asarray(seg[: df.DATA_LEN])[:, 1:])
        label_seg.append(annotation_itpl['cls_idx'][i])
        # extract features
        if feature:
            feature_seg.append(extract_feature(seg))

    # write tf_record file
    if 0 < len(data_seg) == len(label_seg) > 0 and write_tfrecord:
        tfrecord.generate_data(
            path=tfrecord_path + f'/{user_id}-{session_id}_{device_id}_{e4_device_id}.tfrecord',
            dataset=(data_seg, label_seg, feature_seg)
        )

    return data_seg, label_seg, feature_seg


def visualize_model(checkpoint_path, history):
    if history.history.get('accuracy') is not None:
        visualize_path = checkpoint_path + 'visualize'
        if not os.path.exists(visualize_path):
            os.makedirs(visualize_path)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(19.2, 10.8))
        plt.subplot(1, 2, 1)
        plt.plot(range(df.EPOCHS), acc, label='Training Accuracy')
        plt.plot(range(df.EPOCHS), val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(df.EPOCHS), loss, label='Training Loss')
        plt.plot(range(df.EPOCHS), val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(visualize_path + '/visualize.png')
        plt.show()


def main(prepare_data=False):
    data, label, feature = [], [], []
    for user_id in df.USER_ID_TRAIN:
        for session_id in df.SESSION_ID_TRAIN:
            for device_id in df.DEVICE_ID:
                for e4_device_id in df.E4_DEVICE_ID:
                    if prepare_data:
                        data_seg, label_seg, feature_seg = process_data(user_id, session_id, device_id, e4_device_id, df.OPENPACK_VERSION, df.DATASET_ROOTDIR, tfrecord_path=df.TFRECORD_TRAIN_PATH)
                        label_seg = tf.keras.utils.to_categorical(label_seg, df.NUM_CLASSES).astype('int64')

                    else:  # load dat from tfrecord files
                        tfrecord_name = df.TFRECORD_TRAIN_PATH + '/{}-{}_{}_{}.tfrecord'.format(user_id, session_id, device_id, e4_device_id)
                        if not os.path.isfile(tfrecord_name):
                            print('{} no exists!'.format(tfrecord_name))
                            continue

                        print('\n+++++++++++++++++++++++++++++++' +
                              ' load TFRecords: {} '.format(
                                  os.path.basename(tfrecord_name)) +
                              '+++++++++++++++++++++++++++++++')
                        data_seg, feature_seg, label_seg = tfrecord.get_dataset_from_tfrecord(tfrecord_name)
                    data.extend(data_seg)
                    label.extend(label_seg)
                    feature.extend(feature_seg)
    # result = list(map(process_data, args1, args2, args3, args4, args5, args6))

    # region Training
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)

    # train_dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(df.BATCH_SIZE)

    # eval_dataset
    eval_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    eval_dataset = eval_dataset.batch(df.BATCH_SIZE)

    print('\nII. Training Process')
    model = ActivityModel(input_len=df.DATA_LEN, num_data_type=df.NUM_DATA_TYPE, output_shape=df.NUM_CLASSES).make()
    model.summary()

    callbacks, path = CreateCallBack().creating_callbacks(df.SAVE_CKPT_PATH)

    # resume training
    initial_epoch = 0
    resume = tf.train.latest_checkpoint(path[0])
    resume = False
    if resume:
        initial_epoch = int(resume.split("/")[-1][:-5])
        model.load_weights(resume)

    print('\n+++++++++++++++++++++++++++++++ Training Process: +++++++++++++++++++++++++++++++')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=df.LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_dataset,
                        epochs=df.EPOCHS,
                        batch_size=df.BATCH_SIZE,
                        initial_epoch=initial_epoch,
                        verbose=1,
                        validation_data=eval_dataset,
                        use_multiprocessing=False,
                        workers=1,
                        callbacks=callbacks)

    visualize_model(checkpoint_path=df.SAVE_CKPT_PATH, history=history)

    # model_ml = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    # model_ml.fit(X_train, y_train)
    # endregion

    # region Testing
    y_predict = model.predict(np.array(X_test))
    y_predict_convert = np.argmax(y_predict, axis=1)
    y_test_convert = np.argmax(y_test, axis=1)
    print(classification_report(y_test_convert, y_predict_convert))

    ConfusionMatrixDisplay(confusion_matrix(y_test_convert, y_predict_convert)).plot()
    # plt.xticks(rotation=45, ha='right')
    plt.show()
    # endregion


if __name__ == "__main__":
    main()
