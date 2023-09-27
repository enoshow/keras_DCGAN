from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers.core import Flatten
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import glob
import random
import cv2
import glob
import datetime
# import matplotlib.pyplot as plt

n_colors = 1
batch_size = 30


def generator_model():
    model = Sequential()

    model.add(Dense(1024, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(256 *32 * 32)) # 128 : (256*8*8)
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Reshape((32, 32, 256))) # 128 : (8*8*256)

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(n_colors, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=(512, 512, n_colors), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 256x256

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 128x128

    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 64x64

    model.add(Conv2D(512, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 32x32

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def image_batch(batch_size, i):
    files = glob.glob(r"./in_images/**/*.pgm", recursive=True) # original:png
    files = random.sample(files, batch_size)
    # print(files)
    res = []
    for path in files:
        img = Image.open(path)    
        img = img.resize((512, 512)) # defo : 64 * 64
        arr = np.array(img)
        arr = (arr - 127.5) / 127.5
        arr.resize((512, 512, n_colors)) # defo : 64 * 64
        res.append(arr)
    return np.array(res)

def combine_images(generated_images, cols=1, rows=1): # cols = 横, rows = 縦
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]

    if n_colors == 1:  # グレースケールの場合
        image = np.zeros((rows * h, cols * w))
        for index, img in enumerate(generated_images):
            if index >= cols * rows:
                break
            i = index // cols
            j = index % cols
            image[i*h:(i+1)*h, j*w:(j+1)*w] = img[:, :, 0]  # 最後の次元を取り除く
        image = image * 127.5 + 127.5
        image = Image.fromarray(image.astype(np.uint8))
    else:  # カラーの場合
        image = np.zeros((rows * h, cols * w, n_colors))
        for index, img in enumerate(generated_images):
            if index >= cols * rows:
                break
            i = index // cols
            j = index % cols
            image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
        image = image * 127.5 + 127.5
        image = Image.fromarray(image.astype(np.uint8))

    return image

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    discriminator = discriminator_model()
    generator = generator_model()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    set_trainable(discriminator, False)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    print(generator.summary())
    print(discriminator_on_generator.summary())

    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    print(discriminator.summary())

    for i in range(70 * 1000):
        batch_images = image_batch(batch_size, i)

        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        generated_images = generator.predict(noise)
        X = np.concatenate((batch_images, generated_images))
        y = [1] * batch_size + [0] * batch_size
        y=np.array(y)
        d_loss = discriminator.train_on_batch(X, y)
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        batch = [1] * batch_size # batch_sizeの計算をここで行い、
        batch=np.array(batch) # ここでnp配列に変換し、
        g_loss = discriminator_on_generator.train_on_batch(noise, batch) # batchに代入したものをここで使用
        if i % 100 == 0:
            print("\n\n\n\n\n")
            print("step %d d_loss, g_loss : %g %g" % (i, d_loss, g_loss))
            print("\n\n\n\n\n")
            image = combine_images(generated_images)
            os.system('mkdir -p ./gen_images')
            image.save("./gen_images/gen%05d.png" % i)
            # generator.save_weights('generator.h5', True)
            # discriminator.save_weights('discriminator.h5', True)

    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d_%H-%M-%S.%f")
    os.mkdir(f"./save/{date}")

    discriminator_on_generator.save(f"./save/{date}/dis_on_generator.h5")
    discriminator.save(f"./save/{date}/discriminator.h5")

    gen_model = load_model(f'./save/{date}/dis_on_generator.h5')
    gen_model.summary()
    dis_model = load_model(f'./save/{date}/discriminator.h5')
    dis_model.summary()

main()
