from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import datasets.base_new as input_data
import sys
import glob
import os
import numpy as np
from datetime import datetime

data_dir = 'E:\\Python\\base_captcha-tensorflow\\datasets\\images\\char-5-epoch-test\\'
meta, train_data, test_data = input_data.load_data(data_dir, flatten=False)

LABEL = meta['label_choices']
LABEL_SIZE = meta['label_size']
NUM_PER_IMAGE = meta['num_per_image']
IMAGE_HEIGHT = meta['height']
IMAGE_WIDTH = meta['width']
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

print('data loaded')
print('train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0]))
print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))
print('NUM_PER_IMAGE %s' % (NUM_PER_IMAGE))

sess = tf.Session()
tf.keras.backend.set_session(sess)

x = tf.placeholder(tf.float32, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
x_image = tf.reshape(x, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
#y_ = tf.placeholder(tf.float32, [NUM_PER_IMAGE * LABEL_SIZE], )
#print(type(x_image))

#model = load_model('model.h5')
#model.summary()



model = tf.keras.Sequential()

#print(x_image.get_shape())
shape = x_image.get_shape()
#print('input shape need to be ', shape)
#model.add(tf.keras.layers.Conv2D(32, kernel_size=(11, 11), activation='relu', input_shape=shape))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(tf.keras.layers.Conv2D(32, kernel_size=(9, 9), activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#model.add(tf.keras.layers.Conv2D(48, kernel_size=(5, 5), activation='relu'))

#model.add(tf.keras.layers.Conv2D(48, kernel_size=(5, 5), activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))

#model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.25))#0.25
#model.add(tf.keras.layers.Flatten())


model.add(tf.keras.layers.Conv2D(36, (9, 9), padding='same', input_shape=shape, kernel_regularizer=tf.keras.regularizers.l2(0.001)))#32
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(36, (7, 7), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(48, (5, 5), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))#64
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(48, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(56, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))#128
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(56, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(64, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(NUM_PER_IMAGE * LABEL_SIZE * 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(NUM_PER_IMAGE * LABEL_SIZE,  activation='softmax'))

print('outsize', NUM_PER_IMAGE * LABEL_SIZE)
#optAdam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0, amsgrad=False)
optAdam = tf.keras.optimizers.Adam(lr=0.001, clipnorm=1.)

#def captcha_metric(y_true, y_pred):
#   y_pred = tf.keras.backend.reshape(y_pred, (None, LABEL))
#    y_true = tf.keras.backend.reshape(y_true, (None, LABEL))
#    y_p = tf.keras.backend.argmax(y_pred, axis=1)
#    y_t = tf.keras.backend.argmax(y_true, axis=1)
#    r = tf.keras.backend.mean(tf.keras.backend.cast(tf.keras.backend.equal(y_p, y_t), 'float32'))
#    return r


def mean_pred(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))



model.summary()

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optAdam, metrics=[mean_pred])
#'top_k_categorical_accuracy
checkpoint_path = "./checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

logdir = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")


def lr_schedule(epoch):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(epoch)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
           stddev = tf.sqrt(tf.reduce_mean(tf.square(epoch - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(epoch))
        tf.summary.scalar('min', tf.reduce_min(epoch))
        tf.summary.histogram('histogram', epoch)


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
tensorboard_log_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)


#new_model = tf.keras.models.load_model('my_model.h5')
#model.load_weights('./checkpoints/my_checkpoint')

#checkpoint_path = "./checkpoints/training_1/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

################
model.load_weights(checkpoint_path)

print('loaded weights')

#MAX_STEPS = 100
BATCH_SIZE = 50
EPOCHS = 1


LOAD_SIZE = train_data.images.shape[0]-50
#LOAD_SIZE = 100

#batch_xs, batch_ys = train_data.next_batch(LOAD_SIZE)
#batch_xs = batch_xs.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))

#training_history = model.fit(x=batch_xs, y=batch_ys, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.05, callbacks=[cp_callback, tensorboard_log_callback])


#TRAIN_SIZE = 1
#batch_train_xs, batch_train_ys = train_data.next_batch(TRAIN_SIZE)
#batch_train_xs = batch_train_xs.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
#score = model.evaluate(batch_train_xs, batch_train_ys)

#print(type(score))
#print(score)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

#model.save('./model.h5')
#model.save_weights('./checkpoints/my_checkpoint')
model.save(r'./checkpoints/my_model.h5')

#print("Average test loss: ", np.average(training_history.history['loss']))
#print('saved')



TRAIN_SIZE = train_data.images.shape[0]
batch_train_xs, batch_train_ys = train_data.next_batch(TRAIN_SIZE)

batch_train_xs = batch_train_xs.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
predict = model.predict(batch_train_xs, verbose=1, batch_size=TRAIN_SIZE)
#print(predict)
#print(LABEL, type(LABEL))


print(predict.shape)
print(batch_train_ys.shape)

def out_arr(_predict):
    for i in _predict:
        out_str = []
        #print(i)
        for j in range(0, 5):
            #print(j*25, (j+1)*25)
            nuber = i[j*25:(j+1)*25]
            #print(nuber)
            out_str.append(np.argmax(nuber))
        #print('out', out_str)
        for j in out_str:
            print(LABEL[j], end='')
        print('')
out_arr(predict)
print('new era')
out_arr(batch_train_ys)

print('shape', predict.shape, batch_train_ys.shape, len(batch_train_ys), len(predict))
print('get data ', predict)
print('get data ', batch_train_ys)

print('saved')

#callbacks=[tensorboard])
#input_tensor = tf.keras.layers.Input(x_image)
#x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer='he_normal')(x_image)
#x = tf.keras.layers.BatchNormalization(axis=3)(x)
#x = tf.keras.layers.Activation('relu')(x)
#x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer='he_normal')(x)
#x = tf.keras.layers.BatchNormalization(axis=3)(x)
#x = tf.keras.layers.Activation('relu')(x)
#x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer='he_normal')(x)
#x = tf.keras.layers.BatchNormalization(axis=3)(x)
#x = tf.keras.layers.Activation('relu')(x)
#x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), kernel_initializer='he_normal')(x)
#x = tf.keras.layers.BatchNormalization(axis=3)(x)
#x = tf.keras.layers.Activation('relu')(x)
#print(type(x), x.get_shape)
#x = tf.keras.layers.GlobalAveragePooling2D()(x)

#print(type(x), x.get_shape)
#x = tf.keras.layers.Dense(IMAGE_SIZE, activation='softmax', kernel_initializer='he_normal')(x)

#print(type(x_image), x_image.get_shape)
#print(type(x), x.get_shape)
#model = tf.keras.models.Model(x_image, x)

# We'll monitor whether the moving mean and moving variance of the first BatchNorm layer is being updated as it should.
#moving_mean = tf.reduce_mean(model.layers[2].moving_mean)
#moving_variance = tf.reduce_mean(model.layers[2].moving_variance)

#adamOptimiz = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#adamOptimiz = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False)
#model.compile(loss='categorical_crossentropy', optimizer=adamOptimiz)

#model.fit(x_image, y_, batch_size=32, epochs=10)

#model.summary()

#BATCH_SIZE = 100
#batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)
#score = model.evaluate(batch_xs, batch_ys, verbose=0, batch_size=BATCH_SIZE)

#print('Test score:', score[0])
#print('Test accuracy:', score[1])

