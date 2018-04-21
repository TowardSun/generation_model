# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


def glorot_init(shape):
    return tf.trandom_normal(shape=shape, stddev=1.0 / tf.sqrt(shape[0] / 2))


class VAE:

    def __init__(self, image_dim, hidden_dim, latent_dim, sess, lr=0.001, batch_size=64, num_steps=30000):
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps

        self._init_weights()
        self._build_model()
        self._build_decoder_model()

    def _init_weights(self):
        self.weights = {
            'encoder_h1': tf.Variable(glorot_init([self.image_dim, self.hidden_dim])),
            'z_mean': tf.Variable(glorot_init([self.hidden_dim, self.latent_dim])),
            'z_std': tf.Variable(glorot_init([self.hidden_dim, self.latent_dim])),
            'decoder_h1': tf.Variable(glorot_init([self.latent_dim, self.hidden_dim])),
            'decoder_out': tf.Variable(glorot_init([self.hidden_dim, self.image_dim]))
        }

        self.bias = {
            'encoder_b1': tf.Variable(glorot_init([self.hidden_dim])),
            'z_mean': tf.Variable(glorot_init([self.latent_dim])),
            'z_std': tf.Variable(glorot_init([self.latent_dim])),
            'decoder_b1': tf.Variable(glorot_init([self.hidden_dim])),
            'decoder_out': tf.Variable(glorot_init([self.image_dim]))
        }

    def _build_model(self):
        self.input_image = tf.placeholder(tf.float32, shape=[None, self.image_dim])

        with tf.name_scope('encoder'):
            encoder = tf.matmul(self.input_image, self.weights['encoder_h1']) + self.bias['encoder_b1']
            encoder = tf.nn.tanh(encoder)

            z_mean = tf.matmul(encoder, self.weights['z_mean']) + self.bias['z_mean']
            z_std = tf.matmul(encoder, self.weights['z_std']) + self.bias['z_std']

            # Sampler: Normal random distribution
            eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
            self.z = z_mean + tf.exp(z_std / 2) * eps

        with tf.name_scope('decoder'):
            decoder = tf.matmul(self.z, self.weights['decoder_h1']) + self.bias['decoder_b1']
            decoder = tf.nn.tanh(decoder)
            decoder = tf.matmul(decoder, self.weights['decoder_out']) + self.bias['decoder_out']
            self.output = tf.nn.sigmoid(decoder)

        self.loss = self.vae_loss(self.input_image, self.output, z_mean, z_std)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def _build_decoder_model(self):
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.latent_dim])
        decoder = tf.matmul(self.noise_input, self.weights['decoder_h1']) + self.bias['decoder_b1']
        decoder = tf.nn_tanh(decoder)
        decoder = tf.matmul(decoder, self.weights['decoder_out']) + self.bias['decoder_out']
        self.gen_output = tf.nn.sigmoid(decoder)

    @staticmethod
    def vae_loss(x, x_rec, z_mean, z_std):
        with tf.name_scope('vae_loss'):
            rec_loss = x * tf.log(1e-10 + x_rec) + (1 - x) * tf.log(1e-10 + 1 - x_rec)
            rec_loss = -tf.reduce_sum(rec_loss, axis=1)

            # keep variance and add regularization on the encoding representation
            kl_loss = 1 + z_std - tf.exp(z_std) - tf.square(z_mean)
            kl_loss = -0.5 * tf.reduce_sum(kl_loss, 1)

        return tf.reduce_mean(rec_loss + kl_loss)

    def train(self):
        for i in range(1, self.num_steps + 1):
            batch_x, _ = mnist.train.next_batch(self.batch_size)
            _, l = self.sess.run([self.train_op, self.loss], feed_dict={self.input_image: batch_x})

            if i % 100 == 0 or i == 1:
                logger.info('Step %i, train loss: %f' % (i, l))

            if i % 1000 == 0:
                batch_x, _ = mnist.test.next_batch(self.batch_size)
                l = self.sess.run([self.loss], feed_dict={self.input_image: batch_x})
                logger.info('Step %i, test loss: %f' % (i, l))

    def test(self, n=20):
        x_axis = np.linspace(-3, 3, n)
        y_axis = np.linspace(-3, 3, n)

        canvas = np.empty(28 * n, 28 * n)
        for i, xi in enumerate(x_axis):
            for j, yj in enumerate(y_axis):
                z_mu = np.array([[yj, xi]] * self.batch_size)
                x_mean = self.sess.run([self.gen_output], feed_dict={self.noise_input: z_mu})
                canvas[(n - i - 1) * 28: (n - i) * 28, j * 28: (j + 1) * 28] = x_mean[0].reshape(28, 28)
        self.save_plot(canvas)

    @staticmethod
    def save_plot(canvas):
        save_dir = './results'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pdf')
        pp = PdfPages(save_path)
        plt.figure(figsize=(8, 10))

        plt.imshow(canvas, origin='upper', cmap='gray')
        plt.savefig(pp, format='pdf')
        pp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='variational auto encoder')
    parser.add_argument('--image_dim', default=784, type=int, help='input image dimension')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden layer dimension')
    parser.add_argument('--latent_dim', default=2, type=int, help='latent variable dimension')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    init = tf.global_variable_initializer()

    with tf.Session() as run_sess:
        run_sess.run(init)
        vae_model = VAE(
            args.image_dim, args.hidden_dim, args.latent_dim, run_sess, lr=args.learning_rate,
            batch_size=args.batch_size
        )

        vae_model.train()
        vae_model.test()
