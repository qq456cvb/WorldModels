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
import matplotlib.pyplot as plt

STEPS_PER_EPOCH = 100
BATCH_SIZE = 256


class GymDataflow(RNGDataFlow):
    def __init__(self, env):
        self.env = env

    def get_data(self):
        while True:
            self.env.reset()
            i = 0
            while True:
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                img = observation
                img = cv2.resize(img, (64, 64))
                img = img.astype(np.float32) / 255.
                i += 1
                if self.env.env.t < 1.:
                    continue
                yield [img]

                # if i > 3000:
                #     break
                if done:
                    break


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 64, 64, 3], 'state_in')]

    def build_graph(self, img):
        scope = 'AutoEncoder'
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(1e-3)):

                l = slim.conv2d(img, 32, 4, 2, 'VALID')
                l = slim.conv2d(l, 64, 4, 2, 'VALID')
                l = slim.conv2d(l, 128, 4, 2, 'VALID')
                l = slim.conv2d(l, 256, 4, 2, 'VALID')
                l = slim.flatten(l)

                mean = slim.fully_connected(l, 32, None)
                log_var = slim.fully_connected(l, 32, None)
                std_var = tf.exp(log_var / 2)
                encoding = tf.identity(mean, name='encoding')

                # is_training = get_current_tower_context().is_training
                # if not is_training:
                #     return

                z = tf.random_normal(tf.shape(mean)) * std_var + mean
                z = slim.fully_connected(z, 1024)

                l = tf.reshape(z, [-1, 1, 1, 1024])
                l = slim.conv2d_transpose(l, 128, 5, 2, 'VALID')
                l = slim.conv2d_transpose(l, 64, 5, 2, 'VALID')
                l = slim.conv2d_transpose(l, 32, 6, 2, 'VALID')
                l = slim.conv2d_transpose(l, 3, 6, 2, 'VALID', activation_fn=None)

                output = tf.sigmoid(l, name='output')

        kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0, name='kl_loss')
        # gaussian mse loss
        reconstruct_loss = tf.reduce_sum(tf.square(img - output), axis=[1, 2, 3])
        # entropy loss
        # reconstruct_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=img, logits=l), axis=[1, 2, 3])
        reconstruct_loss = tf.reduce_mean(reconstruct_loss, name='reconstruct_loss')
        l2_loss = tf.identity(regularize_cost_from_collection(), name='l2_loss')
        add_moving_summary(reconstruct_loss)
        add_moving_summary(l2_loss)
        add_moving_summary(kl_loss)
        loss = reconstruct_loss + kl_loss
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        opt = tf.train.GradientDescentOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5))]
        # SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


def train():
    dirname = os.path.join('train_log', 'auto_encoder')
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

    dataflow = GymDataflow(gym.make('CarRacing-v0'))
    if os.name == 'nt':
        dataflow = PrefetchData(dataflow, nr_proc=multiprocessing.cpu_count() // 2,
                                nr_prefetch=multiprocessing.cpu_count() // 2)
    else:
        dataflow = PrefetchDataZMQ(dataflow, nr_proc=multiprocessing.cpu_count() // 2)
    dataflow = BatchData(dataflow, BATCH_SIZE)
    config = AutoResumeTrainConfig(
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
        max_epoch=200,
    )
    trainer = AsyncMultiGPUTrainer(train_tower) if nr_gpu > 1 else SimpleTrainer()
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    # train()
    env = gym.make('CarRacing-v0')
    pred = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader('train_log/auto_encoder/checkpoint'),
        input_names=['state_in'],
        output_names=['AutoEncoder/output']
    ))
    while True:
        env.reset()
        while True:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if env.env.t < 1.:
                continue
            img = observation
            cv2.imshow('origin', img)
            # cv2.waitKey(30)
            img = cv2.resize(img, (64, 64))
            img = img.astype(np.float32) / 255.
            reconstruction = pred([img[None, :, :, :]])[0][0]
            # plt.imshow(img)
            # plt.imshow(reconstruction)
            cv2.imshow('recons', reconstruction)
            cv2.waitKey(10)
            # plt.show()
            if done:
                break
    # # env.reset()
    # # while True:
    # #     action = env.action_space.sample()
    # #     observation, reward, done, info = env.step(action)
    # #     cv2.imshow('test', observation)
    # #     cv2.waitKey(30)
    # model = Model()
    # state_in = model.inputs()[0]
    # model.build_graph(state_in)
    # out = tf.get_default_graph().get_tensor_by_name('AutoEncoder/output:0')
    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, 'train_log/auto_encoder/model-200')
    #     while True:
    #         env.reset()
    #         while True:
    #             action = env.action_space.sample()
    #             observation, reward, done, info = env.step(action)
    #             img = observation
    #             cv2.imshow('origin', img)
    #             # cv2.waitKey(30)
    #             img = cv2.resize(img, (64, 64))
    #             img = img.astype(np.float32) / 255.
    #             reconstruction = sess.run(out, feed_dict={state_in: img.reshape(1, 64, 64, 3)})[0]
    #             # plt.imshow(img)
    #             # plt.imshow(reconstruction)
    #             cv2.imshow('recons', reconstruction)
    #             cv2.waitKey()
    #             # plt.show()
    #             if done:
    #                 break

    # from tensorflow.python.client import device_lib
    #
    # print(device_lib.list_local_devices())
    # env = gym.make('CarRacing-v0')
    # df = GymDataflow(env)
    # for _ in df.get_data():
    #     pass
