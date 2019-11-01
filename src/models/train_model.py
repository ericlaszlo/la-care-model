import joblib
import os
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
tf.keras.backend.set_floatx('float64')

LOSS_OBJECT = tf.keras.losses.SparseCategoricalCrossentropy()
OPTIMIZER = tf.keras.optimizers.Adam()

TRAIN_LOSS = tf.keras.metrics.Mean(name='TRAIN_LOSS')
TRAIN_ACCURACY = tf.keras.metrics.SparseCategoricalAccuracy(
    name='TRAIN_ACCURACY')

TEST_LOSS = tf.keras.metrics.Mean(name='TEST_LOSS')
TEST_ACCURACY = tf.keras.metrics.SparseCategoricalAccuracy(
    name='TEST_ACCURACY')


EPOCHS = 20
BATCH_SIZE = 128


def make_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5),
                    padding='same',
                    activation='relu',
                    input_shape=[28, 28, 1]))
    model.add(Conv2D(32, (5,5),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3),
                    padding='same',
                    activation='relu'))
    model.add(Conv2D(64, (3,3),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),
                           strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model



MODEL = make_cnn_model()


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = MODEL(images)
        loss = LOSS_OBJECT(labels, predictions)
    gradients = tape.gradient(loss, MODEL.trainable_variables)
    OPTIMIZER.apply_gradients(
        zip(gradients, MODEL.trainable_variables))

    TRAIN_LOSS(loss)
    TRAIN_ACCURACY(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = MODEL(images)
    t_loss = LOSS_OBJECT(labels, predictions)

    TEST_LOSS(t_loss)
    TEST_ACCURACY(labels, predictions)


if __name__ == '__main__':

    current_path = os.getcwd()
    file_name = '/data/processed/mnist.gz'
    (x_train, y_train), (x_test, y_test) = \
            joblib.load(current_path + file_name)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              TRAIN_LOSS.result(),
                              TRAIN_ACCURACY.result()*100,
                              TEST_LOSS.result(),
                              TEST_ACCURACY.result()*100))

        TRAIN_LOSS.reset_states()
        TRAIN_ACCURACY.reset_states()
        TEST_LOSS.reset_states()

        TEST_ACCURACY.reset_states()

    MODEL.save(current_path + '/models/cnn',
              save_format='tf')
