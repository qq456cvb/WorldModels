from tensorpack.tfutils.summary import add_moving_summary
from tensorpack import *
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.tfutils import (
    get_current_tower_context, optimizer)
from tensorpack.utils.gpu import get_nr_gpu
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import sys
import os
import multiprocessing
import tensorpack.dataflow
import tensorflow as tf
import numpy as np
import gym
import cv2
import tensorflow.contrib.rnn as rnn

STEPS_PER_EPOCH = 100
BATCH_SIZE = 8
ACTION_DIM = 4
Z_DIM = 32
SEQ_LEN = 20
MIX_GAUSSIANS = 5


class RNNDataflow(RNGDataFlow):
    def __init__(self, env):
        self.env = env

    def get_data(self):
        while True:
            self.env.reset()
            while True:
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                # img = cv2.resize(observation, (64, 64))
                # img = img.astype(np.float32) / 255.
                if done:
                    yield [np.zeros([SEQ_LEN], dtype=np.int32),
                           np.zeros([SEQ_LEN, Z_DIM], dtype=np.float32),
                           np.zeros([SEQ_LEN, Z_DIM], dtype=np.float32)]
                    break


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.int32, [None, SEQ_LEN], 'action_input'),
                tf.placeholder(tf.float32, [None, SEQ_LEN, Z_DIM], 'z_input'),
                tf.placeholder(tf.float32, [None, SEQ_LEN, Z_DIM], 'z_target')]

    def build_graph(self, action, z_in, z_target):
        scope = 'MDN-RNN'
        with tf.variable_scope(scope):
            hidden_dim = 256
            cell = rnn.BasicLSTMCell(hidden_dim)
            init_state = rnn.LSTMStateTuple(tf.placeholder_with_default(tf.zeros(tf.stack([tf.shape(action)[0], hidden_dim]), name='cz'),
                                                     shape=[None, hidden_dim], name='c'),
                                            tf.placeholder_with_default(
                                                tf.zeros(tf.stack([tf.shape(action)[0], hidden_dim]), name='hz'),
                                                shape=[None, hidden_dim], name='h'))
            x = tf.concat([tf.one_hot(action, ACTION_DIM), z_in], axis=-1)
            x_list = tf.unstack(x, axis=1)

            outputs, last_state = rnn.static_rnn(cell, x_list, init_state)
            last_state = tf.identity('last_state')

            # is_training = get_current_tower_context().is_training
            # if not is_training:
            #     return

            # B * S * H
            outputs = tf.stack(outputs, axis=1)
            # outputs = tf.Print(outputs, [tf.shape(outputs)], summarize=10)
            dense = slim.fully_connected(tf.reshape(outputs, [-1, hidden_dim]), 2 * Z_DIM * MIX_GAUSSIANS + MIX_GAUSSIANS, activation_fn=None)
            mean = tf.reshape(dense[:, :Z_DIM * MIX_GAUSSIANS], [-1, MIX_GAUSSIANS, Z_DIM])
            sigma = tf.reshape(tf.exp(dense[:, Z_DIM * MIX_GAUSSIANS:2*Z_DIM * MIX_GAUSSIANS]), [-1, MIX_GAUSSIANS, Z_DIM])
            pi = tf.nn.softmax(dense[:, -MIX_GAUSSIANS:])

            # sample from mixture of gaussian
            z = tf.reshape(tf.tile(z_target, [1, 1, MIX_GAUSSIANS]), [-1, MIX_GAUSSIANS, Z_DIM])
            z = tf.Print(z, [tf.shape(z), tf.shape(mean)], summarize=10)
            # a const positive scalar (2pi)^(-D/2) is omitted
            probs = tf.expand_dims(pi, axis=-1) * tf.exp(-tf.square(z - mean) / (2 * tf.square(sigma) + 1e-8)) / (sigma + 1e-8)

            # (B * S) * Z
            probs = tf.reduce_sum(probs, axis=1)

        loss = tf.reduce_sum(-tf.log(probs), name='loss')
        add_moving_summary(loss)

        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
        # SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


def train():
    dirname = os.path.join('train_log', 'mdn-rnn')
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    nr_gpu = get_nr_gpu()
    if nr_gpu > 0:
        train_tower = list(range(nr_gpu)) or [0]
        logger.info("[Batch-SL] Train on gpu {}".format(
            ','.join(map(str, train_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        train_tower = [0], [0]

    dataflow = RNNDataflow(gym.make('CarRacing-v0'))
    if os.name == 'nt':
        dataflow = PrefetchData(dataflow, nr_proc=multiprocessing.cpu_count() // 2,
                                nr_prefetch=multiprocessing.cpu_count() // 2)
    else:
        dataflow = PrefetchDataZMQ(dataflow, nr_proc=multiprocessing.cpu_count() // 2)
    dataflow = BatchData(dataflow, BATCH_SIZE)
    config = TrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            EstimatedTimeLeft(),
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=100,
    )
    trainer = AsyncMultiGPUTrainer(train_tower) if nr_gpu > 1 else SimpleTrainer()
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    train()
