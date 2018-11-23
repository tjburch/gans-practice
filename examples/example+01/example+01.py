# replicating tutorial here:
# https://www.datacamp.com/community/tutorials/generative-adversarial-networks

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

# Turn on tensorflow
os.environ["KERAS_BACKEND"] = 'tensorflow'
# For reproducibility (can comment out later for validation)
np.random.seed(10)
random_dim = 100

# Let there be many plots open
plt.rcParams.update({'figure.max_open_warning':0})


def load_mnist():
    # outright load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize inputs to [-1, 1]
    x_train = (x_train.astype(np.float32) -127.5)/127.5
    # x_train is (60000, 28, 28) -> (60000, 784) by flattening x/y
    x_train = x_train.reshape(60000,784)
    return (x_train, y_train, x_test, y_test)


# Using adaptive moment estimation (Adam)
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def get_generator(optimizer):
    # build generative model
    generator = Sequential()
    # Input Layer
    generator.add(Dense(256,\
                        input_dim=random_dim, \
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    # Hidden layer
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    # Hidden layer
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))


    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def get_discriminator(optimizer):
    # build generative model
    discriminator = Sequential()
    # Input Layer, same dimensionality
    discriminator.add(Dense(1024, \
                            input_dim=784, \
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    # Hidden layer
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    # Hidden layer
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def build_gan(discriminator, random_dim, generator, optimizer):
    # only train discriminator or generator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator
    gan_output = discriminator(x) # some probability if image is real
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan


# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        # Plots images
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    # Save figure locally (add dir to .gitignore)
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
    #plt.close()


def train(epochs=1, batch_size=128):
    x_train, y_train, x_test, y_test = load_mnist()

    # Split into batches
    batch_count = x_train.shape[0] / batch_size

    # construct the gan
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = build_gan(discriminator, random_dim, generator, adam)

    # iterate over epochs
    for e in xrange(1, epochs+1):
        # Print some updates
        print '-'*15, 'Epoch %d' % e, '-'*15
        for _ in tqdm(xrange(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            # Add fake MNIST to actual MNIST
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            # Train on this batch
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

            # Plot every once in a while
            if e == 1 or e % 20 == 0:
                plot_generated_images(e, generator)


if __name__ == '__main__':
    train(400, 128)
