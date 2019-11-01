import joblib
import os
import tensorflow as tf

if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    current_path = os.getcwd()
    file_name = '/data/processed/mnist.gz'
    joblib.dump(((x_train, y_train), (x_test, y_test)),
                current_path + file_name,
               compress=True)
