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
# model = ActivityModel(input_len=df.DATA_LEN, num_data_type=df.NUM_DATA_TYPE, output_shape=df.NUM_CLASSES).make()
# model.summary()