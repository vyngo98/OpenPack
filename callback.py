import os
import tensorflow as tf


class CreateCallBack:
    def __init__(self):
        super(CreateCallBack, self).__init__()
        # self.datapath = datapath
        # self.dataset = dataset
        # self.predict_dataset = predict_dataset

        # self.epoch_range = range(EPOCHS)
        # self.process = ["train", "test", "eval"]
        self.callbacks = ["total_ckpt", "best_acc", "best_loss", "logs"]
        self.monitor = ["val_accuracy", "val_loss"]

    def creating_callbacks(self, checkpoint_path):
        path = []
        callbacks = []
        for index in range(len(self.callbacks)):
            ckpt_path = '{}/{}/'.format(checkpoint_path, self.callbacks[index])
            path.append(ckpt_path)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)

            if index < (len(self.callbacks) - 1):
                cb_path = '{}/{}'.format(ckpt_path, "{epoch:04d}.ckpt")

                if index > 0:
                    monitor = self.monitor[index - 1]
                    save_best_only = True
                else:
                    monitor = self.monitor[index]
                    save_best_only = False

                cb = tf.keras.callbacks.ModelCheckpoint(filepath=cb_path,
                                                        monitor=monitor,
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        save_best_only=save_best_only)
            else:
                cb = tf.keras.callbacks.TensorBoard(log_dir=ckpt_path)

            callbacks.append(cb)

        return callbacks, path