import define as df
from processing_data import *
import tfrecord as tfrecord

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


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
        data_seg.append(np.asarray(seg[: df.DATA_LEN]))
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
        exit()

    return data_seg, label_seg, feature_seg


def main(prepare_data=False):
    data, label, feature = [], [], []
    for user_id in df.USER_ID_TRAIN:
        for session_id in df.SESSION_ID_TRAIN:
            for device_id in df.DEVICE_ID:
                for e4_device_id in df.E4_DEVICE_ID:
                    if prepare_data:
                        data_seg, label_seg, feature_seg = process_data(user_id, session_id, device_id, e4_device_id, df.OPENPACK_VERSION, df.DATASET_ROOTDIR, tfrecord_path=df.TFRECORD_TRAIN_PATH)

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
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=42)

    model_ml = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model_ml.fit(X_train, y_train)
    # endregion

    # region Testing
    y_predict = model_ml.predict(X_test)
    print(classification_report(y_test, y_predict))
    # confusion_matrix(y_test, y_predict)
    plot_confusion_matrix(model_ml, X_test, y_test)
    plt.show()
    # endregion


if __name__ == "__main__":
    main()
