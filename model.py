from define import *
import tensorflow as tf


class ActivityModel:
    def __init__(self, input_len, num_data_type, output_shape):
        super(ActivityModel, self).__init__()
        self.input_len = input_len
        self.num_data_type = num_data_type
        self.output_shape = output_shape

    def make(self):
        model_input = tf.keras.layers.Input(shape=(self.input_len, self.num_data_type))
        layer = model_input

        layer = tf.keras.layers.Conv1D(16, 3, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        layer = tf.keras.layers.Conv1D(32, 3, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        layer = tf.keras.layers.Conv1D(32, 3, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(units=64, activation='relu')(layer)
        layer = tf.keras.layers.Dense(units=32, activation='relu')(layer)
        layer = tf.keras.layers.Dense(self.output_shape, activation="softmax")(layer)
        # layer = tf.expand_dims(layer, axis=1)
        model_output = layer
        model = tf.keras.Model(model_input, model_output)

        # Compile is option
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                'accuracy'
            ]
        )
        return model


class ActivityModel2:
    def __init__(self, input_len, num_data_type, output_shape):
        super(ActivityModel2, self).__init__()
        self.input_len = input_len
        self.num_data_type = num_data_type
        self.output_shape = output_shape

    def make(self):
        model_input = tf.keras.layers.Input(shape=(self.input_len, self.num_data_type))
        layer = model_input

        layer = tf.keras.layers.Conv1D(196, 12, activation='relu')(layer)
        # layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=4)(layer)
        # layer = tf.keras.layers.Conv1D(32, 3, activation='relu')(layer)
        # layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        # layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        # layer = tf.keras.layers.Conv1D(32, 3, activation='relu')(layer)
        # layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        # layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(units=1024, activation='relu')(layer)
        # layer = tf.keras.layers.Dense(units=32, activation='relu')(layer)
        layer = tf.keras.layers.Dense(self.output_shape, activation="softmax")(layer)

        model_output = layer
        model = tf.keras.Model(model_input, model_output)

        # Compile is option
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                'accuracy'
            ]
        )
        return model


# import define as df
#
#
# model = ActivityModel(input_len=df.WINDOW_SIZE*df.FS_TARGET, num_data_type=df.NUM_DATA_TYPE, output_shape=df.NUM_CLASSES).make()
# model.summary()

# model = ActivityModel2(input_len=df.DATA_LEN, num_data_type=df.NUM_DATA_TYPE, output_shape=df.NUM_CLASSES).make()
# model.summary()


class ActivityModel3:
    def __init__(self, imu_len, kinect_len, e4_len, bvp_len, num_imu_ch, num_kinect_ch, num_e4_ch, num_bvp_ch,
                 output_shape):
        super(ActivityModel3, self).__init__()
        self.imu_len = imu_len
        self.kinect_len = kinect_len
        self.e4_len = e4_len
        self.bvp_len = bvp_len
        self.num_imu_ch = num_imu_ch
        self.num_kinect_ch = num_kinect_ch
        self.num_e4_ch = num_e4_ch
        self.num_bvp_ch = num_bvp_ch
        self.output_shape = output_shape

    def make(self):
        imu_input = tf.keras.layers.Input(shape=(self.imu_len, self.num_imu_ch))
        kinect_input = tf.keras.layers.Input(shape=(self.kinect_len, self.num_kinect_ch))
        e4_input = tf.keras.layers.Input(shape=(self.e4_len, self.num_e4_ch))
        bvp_input = tf.keras.layers.Input(shape=(self.bvp_len, self.num_bvp_ch))

        layer_imu = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(imu_input)
        layer_imu = tf.keras.layers.Dropout(rate=0.1)(layer_imu)
        layer_imu = tf.keras.layers.MaxPool1D(pool_size=2)(layer_imu)

        layer_kinect = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(
            kinect_input)
        layer = tf.keras.layers.Concatenate()([layer_imu, layer_kinect])

        layer = tf.keras.layers.Conv1D(64, 7, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.1)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer = tf.keras.layers.Conv1D(64, 3, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer = tf.keras.layers.Conv1D(64, 1, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.5)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(units=512, activation='relu')(layer)
        layer = tf.keras.layers.Dense(self.output_shape, activation="softmax")(layer)

        model_output = layer
        model = tf.keras.Model([imu_input, kinect_input], model_output)

        # Compile is option
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                'accuracy'
            ]
        )
        return model


class ActivityModel4:
    def __init__(self, imu_len, kinect_len, e4_len, bvp_len, num_imu_ch, num_kinect_ch, num_e4_ch, num_bvp_ch,
                 output_shape):
        super(ActivityModel4, self).__init__()
        self.imu_len = imu_len
        self.kinect_len = kinect_len
        self.e4_len = e4_len
        self.bvp_len = bvp_len
        self.num_imu_ch = num_imu_ch
        self.num_kinect_ch = num_kinect_ch
        self.num_e4_ch = num_e4_ch
        self.num_bvp_ch = num_bvp_ch
        self.output_shape = output_shape

    def make(self):
        imu_input = tf.keras.layers.Input(shape=(self.imu_len, self.num_imu_ch))
        kinect_input = tf.keras.layers.Input(shape=(self.kinect_len, self.num_kinect_ch))
        e4_input = tf.keras.layers.Input(shape=(self.e4_len, self.num_e4_ch))
        bvp_input = tf.keras.layers.Input(shape=(self.bvp_len, self.num_bvp_ch))

        layer_imu = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(imu_input)
        layer_imu = tf.keras.layers.Dropout(rate=0.1)(layer_imu)
        layer_imu = tf.keras.layers.MaxPool1D(pool_size=2)(layer_imu)

        layer_kinect = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(
            kinect_input)

        layer_bvp = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(bvp_input)
        layer_bvp = tf.keras.layers.Dropout(rate=0.1)(layer_bvp)
        layer_bvp = tf.keras.layers.MaxPool1D(pool_size=2)(layer_bvp)
        layer_bvp = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(layer_bvp)
        layer_bvp = tf.keras.layers.Dropout(rate=0.1)(layer_bvp)
        layer_bvp = tf.keras.layers.MaxPool1D(pool_size=2)(layer_bvp)

        layer = tf.keras.layers.Concatenate()([layer_imu, layer_kinect, layer_bvp])

        layer = tf.keras.layers.Conv1D(64, 7, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.1)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer_e4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(e4_input)
        layer_e4 = tf.keras.layers.Dropout(rate=0.1)(layer_e4)
        layer = tf.keras.layers.Concatenate()([layer, layer_e4])

        layer = tf.keras.layers.Conv1D(64, 1, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(rate=0.5)(layer)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(units=512, activation='relu')(layer)
        layer = tf.keras.layers.Dense(self.output_shape, activation="softmax")(layer)

        model_output = layer
        model = tf.keras.Model([imu_input, kinect_input, e4_input, bvp_input], model_output)

        # Compile is option
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                'accuracy'
            ]
        )
        return model


class ActivityModel5:
    def __init__(self, imu_len, kinect_len, e4_len, bvp_len, num_imu_ch, num_kinect_ch, num_e4_ch, num_bvp_ch,
                 output_shape):
        super(ActivityModel5, self).__init__()
        self.imu_len = imu_len
        self.kinect_len = kinect_len
        self.e4_len = e4_len
        self.bvp_len = bvp_len
        self.num_imu_ch = num_imu_ch
        self.num_kinect_ch = num_kinect_ch
        self.num_e4_ch = num_e4_ch
        self.num_bvp_ch = num_bvp_ch
        self.output_shape = output_shape

    def conv_with_Batch_Normalisation(self, prev_layer, n_filters, kernels_size):
        x = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernels_size, padding='same')(prev_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='elu')(x)
        return x

    def inception_like(self, prev_layer, n_filters):
        branch_1 = self.conv_with_Batch_Normalisation(prev_layer=prev_layer, n_filters=64, kernels_size=1)

        branch_2 = self.conv_with_Batch_Normalisation(prev_layer=prev_layer, n_filters=64, kernels_size=1)
        branch_2 = self.conv_with_Batch_Normalisation(prev_layer=branch_2, n_filters=94, kernels_size=3)

        branch_3 = self.conv_with_Batch_Normalisation(prev_layer=prev_layer, n_filters=48, kernels_size=1)
        branch_3 = self.conv_with_Batch_Normalisation(prev_layer=branch_3, n_filters=64, kernels_size=5)

        branch_4 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
        branch_4 = self.conv_with_Batch_Normalisation(prev_layer=branch_4, n_filters=n_filters, kernels_size=1)
        layer = tf.keras.layers.Concatenate()([branch_1, branch_2, branch_3, branch_4])
        return layer

    def make(self):
        imu_input = tf.keras.layers.Input(shape=(self.imu_len, self.num_imu_ch))
        kinect_input = tf.keras.layers.Input(shape=(self.kinect_len, self.num_kinect_ch))
        e4_input = tf.keras.layers.Input(shape=(self.e4_len, self.num_e4_ch))
        bvp_input = tf.keras.layers.Input(shape=(self.bvp_len, self.num_bvp_ch))

        layer_e4 = self.conv_with_Batch_Normalisation(e4_input, 1, 1)
        layer_e4 = tf.keras.layers.Flatten()(layer_e4)

        layer_imu = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(imu_input)
        layer_imu = tf.keras.layers.Dropout(rate=0.1)(layer_imu)
        layer_imu = tf.keras.layers.MaxPool1D(pool_size=2)(layer_imu)

        # layer_kinect = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(
        #     kinect_input)
        layer_kinect = kinect_input

        layer_bvp = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(bvp_input)
        layer_bvp = tf.keras.layers.Dropout(rate=0.1)(layer_bvp)
        layer_bvp = tf.keras.layers.MaxPool1D(pool_size=2)(layer_bvp)
        layer_bvp = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(layer_bvp)
        layer_bvp = tf.keras.layers.Dropout(rate=0.1)(layer_bvp)
        layer_bvp = tf.keras.layers.MaxPool1D(pool_size=2)(layer_bvp)

        layer = tf.keras.layers.Concatenate()([layer_imu, layer_kinect, layer_bvp])

        layer = self.inception_like(layer, n_filters=32)
        layer = self.inception_like(layer, n_filters=64)
        layer = self.inception_like(layer, n_filters=64)
        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        layer = self.inception_like(layer, n_filters=128)

        layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        layer = tf.keras.layers.GRU(units=128, return_sequences=True)(layer)
        layer = tf.keras.layers.GRU(units=64, return_sequences=True)(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Concatenate()([layer, layer_e4])
        layer = tf.keras.layers.Dense(units=512, activation='elu')(layer)
        layer = tf.keras.layers.Dense(self.output_shape, activation="softmax")(layer)
        # layer_1 = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding='same')(layer)
        # layer_2 = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')(layer)
        # layer_2 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(layer_2)
        #
        # layer_3 = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')(layer)
        # layer_3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(layer_3)
        #
        # layer_4 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(layer)
        # layer_4 = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')(layer_4)
        # layer = tf.keras.layers.Concatenate()([layer_1, layer_2, layer_3, layer_4])
        # a=0
        # layer = tf.keras.layers.Conv1D(64, 7, activation='relu')(layer)
        # layer = tf.keras.layers.Dropout(rate=0.1)(layer)
        # layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        #
        # layer = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(layer)
        # layer = tf.keras.layers.Dropout(rate=0.25)(layer)
        # layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        #
        # layer_e4 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(e4_input)
        # layer_e4 = tf.keras.layers.Dropout(rate=0.1)(layer_e4)
        # layer = tf.keras.layers.Concatenate()([layer, layer_e4])

        # layer = tf.keras.layers.Conv1D(64, 1, activation='relu')(layer)
        # layer = tf.keras.layers.Dropout(rate=0.5)(layer)
        # layer = tf.keras.layers.MaxPool1D(pool_size=2)(layer)
        #
        # layer = tf.keras.layers.Flatten()(layer)
        # layer = tf.keras.layers.Dense(units=512, activation='relu')(layer)
        # layer = tf.keras.layers.Dense(self.output_shape, activation="softmax")(layer)

        model_output = layer
        model = tf.keras.Model([imu_input, kinect_input, e4_input, bvp_input], model_output)

        # Compile is option
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                'accuracy'
            ]
        )
        return model


# window_size = 2
# model = ActivityModel5(imu_len=FS_TARGET * window_size,
#                        kinect_len=FS_KEYPOINT * window_size,
#                        e4_len=FS_E4 * window_size,
#                        bvp_len=FS_BVP * window_size,
#                        num_imu_ch=10,
#                        num_kinect_ch=17,
#                        num_e4_ch=2,
#                        num_bvp_ch=1,
#                        output_shape=NUM_CLASSES).make()
# model.summary()
