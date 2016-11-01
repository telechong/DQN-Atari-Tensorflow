# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------
import os
import time
import datetime
import random
from collections import deque

import tensorflow as tf
import numpy as np

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.95  # decay rate of past observations
OBSERVE = 50000.  # timesteps to observe before training
EXPLORE = 1000000.  # frames over which to anneal _epsilon
FINAL_EPSILON = 0.1  # 0.001 # final value of _epsilon
INITIAL_EPSILON = 1.0  # 0.01 # starting value of _epsilon
REPLAY_MEMORY = 1000000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 10000

class BrainDQN:
    def __init__(self, init_observation, actions, checkpointpath=None, summarypath=None):
        self._replay_memory = deque()

        # Counters
        self._timestep = 0
        self._prev_timestep = 0
        self._prev_timestamp = time.clock()

        self._fps = tf.Variable(initial_value=0.0, dtype=tf.float32, name='frames_per_second')
        self._meta_state = tf.Variable(initial_value=0, dtype=tf.uint8, name='meta_state')  # (0-observe, 1-explore, 2-train)
        self._loss_function = tf.Variable(initial_value=0.0, dtype=tf.float32, name='loss')

        self._epsilon = tf.Variable(initial_value=INITIAL_EPSILON, dtype=tf.float32, name='epsilon')
        self._actions = actions

        # init Q network
        self._stateinput, \
        self._qvalue, \
        self._w_conv1, \
        self._b_conv_1, \
        self._w_conv_2, \
        self._b_conv2, \
        self._w_conv_3, \
        self._b_conv3, \
        self._w_fc_1, \
        self._b_fc1, \
        self._w_fc_2, \
        self._b_fc2 = self._create_q_network()

        # init Target Q Network
        self._stateinput_t, \
        self._qvalue_t, \
        self._w_conv_1_t, \
        self._b_conv_1_t, \
        self._w_conv_2_t, \
        self._b_conv_2_t, \
        self._w_conv_3_t, \
        self._b_conv_3_t, \
        self._w_fc_1_t, \
        self._b_fc_1_t, \
        self._w_fc_2_t, \
        self._b_fc_2_t = self._create_q_network()

        with tf.name_scope('copy_target_qnetwork'):
            self._copy_target_qnetwork_operation = [self._w_conv_1_t.assign(self._w_conv1),
                                                    self._b_conv_1_t.assign(self._b_conv_1),
                                                    self._w_conv_2_t.assign(self._w_conv_2),
                                                    self._b_conv_2_t.assign(self._b_conv2),
                                                    self._w_conv_3_t.assign(self._w_conv_3),
                                                    self._b_conv_3_t.assign(self._b_conv3),
                                                    self._w_fc_1_t.assign(self._w_fc_1),
                                                    self._b_fc_1_t.assign(self._b_fc1),
                                                    self._w_fc_2_t.assign(self._w_fc_2),
                                                    self._b_fc_2_t.assign(self._b_fc2)]
        with tf.name_scope('train_qnetwork'):
            self._action_input = tf.placeholder("float", [None, self._actions])
            self._y_input = tf.placeholder("float", [None])
            q_action = tf.reduce_sum(tf.mul(self._qvalue, self._action_input), reduction_indices=1, name='q_action')
            self._cost = tf.reduce_mean(tf.square(self._y_input - q_action), name='loss_function')
            self._train_step = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self._cost)

        self._session = tf.InteractiveSession()

        if summarypath is not None:
            tf.scalar_summary(r'overview/meta_state (0-observe, 1-explore, 2-train)', self._meta_state)
            tf.scalar_summary(r'overview/cost (loss function)', self._loss_function)
            tf.scalar_summary(r'overview/epsilon (exploration probability)', self._epsilon)
            tf.scalar_summary(r'performance/frames_per_second', self._fps)
            self._summaries = tf.merge_all_summaries()
            self._summarywriter = tf.train.SummaryWriter(os.path.join(summarypath, str(datetime.datetime.now())), self._session.graph)

        # saving and loading networks
        self._saver = tf.train.Saver()
        self._session.run(tf.initialize_all_variables())

        if checkpointpath is not None:
            try:
                os.makedirs(checkpointpath)
            except OSError:
                pass

            self._checkpoint = tf.train.get_checkpoint_state(checkpointpath)
            if self._checkpoint and self._checkpoint.model_checkpoint_path:
                self._saver.restore(self._session, self._checkpoint.model_checkpoint_path)
                print "Successfully loaded:", self._checkpoint.model_checkpoint_path
            else:
                print "Could not find old network weights"

        self._current_state = np.stack((init_observation, init_observation, init_observation, init_observation), axis=2)

    def set_perception(self, next_observation, action, reward, terminal):
        new_state = np.append(next_observation, self._current_state[:, :, 1:], axis=2)
        self._replay_memory.append((self._current_state, action, reward, new_state, terminal))
        if len(self._replay_memory) > REPLAY_MEMORY:
            self._replay_memory.popleft()
        if self._timestep > OBSERVE:
            self._train_qnetwork()

        if self._timestep % 100 == 0:
            self._write_counters()

        self._current_state = new_state
        self._timestep += 1

    def get_action(self):
        qvalue = self._qvalue.eval(feed_dict={self._stateinput:[self._current_state]})[0]
        action = np.zeros(self._actions)
        action_index = 0
        epsilon = self._epsilon.eval()
        if self._timestep % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(self._actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(qvalue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if epsilon > FINAL_EPSILON and self._timestep > OBSERVE:
            self._epsilon.assign_sub((INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE).op.run()
        return action

    def _create_q_network(self):
        # network weights
        w_conv1 = self._weight_variable([8, 8, 4, 32])
        b_conv1 = self._bias_variable([32])

        w_conv2 = self._weight_variable([4, 4, 32, 64])
        b_conv2 = self._bias_variable([64])

        w_conv3 = self._weight_variable([3, 3, 64, 64])
        b_conv3 = self._bias_variable([64])

        w_fc1 = self._weight_variable([3136, 512])
        b_fc1 = self._bias_variable([512])

        w_fc2 = self._weight_variable([512, self._actions])
        b_fc2 = self._bias_variable([self._actions])

        # input layer
        stateinput = tf.placeholder("float", [None, 84, 84, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self._conv2d(stateinput, w_conv1, 4) + b_conv1)
        # h_pool1 = self._max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, w_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(self._conv2d(h_conv2, w_conv3, 1) + b_conv3)
        h_conv3_shape = h_conv3.get_shape().as_list()
        print "dimension:", h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]
        h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

        # Q Value layer
        qvalue = tf.matmul(h_fc1, w_fc2) + b_fc2

        return stateinput, qvalue, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2

    def _copy_target_qnetwork(self):
        self._session.run(self._copy_target_qnetwork_operation)

    def _train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self._replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        qvalue_batch = self._qvalue_t.eval(feed_dict={self._stateinput_t:next_state_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(qvalue_batch[i]))

        if self._timestep == OBSERVE+1:
            run_metadata = tf.RunMetadata()
            self._session.run(self._train_step,
                              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                              run_metadata=run_metadata,
                              feed_dict={self._y_input : y_batch,
                                         self._action_input : action_batch,
                                         self._stateinput : state_batch})
            self._summarywriter.add_run_metadata(run_metadata, 'step_%d' % self._timestep)
        else:
            self._train_step.run(feed_dict={self._y_input : y_batch,
                                            self._action_input : action_batch,
                                            self._stateinput : state_batch})

        # calculate loss function (we do it via proxy variable since summaries are run separatly)
        value = self._cost.eval(feed_dict={self._y_input : y_batch,
                                           self._action_input : action_batch,
                                           self._stateinput : state_batch})
        self._loss_function.assign(value).op.run()

        # save network every 100000 iteration
        if self._timestep % 100000 == 0:
            self._saver.save(self._session, 'dqn', global_step=self._timestep)

        if self._timestep % UPDATE_TIME == 0:
            self._copy_target_qnetwork()

    def _write_counters(self):
        # FPS
        new_timestamp = time.clock()
        self._fps.assign((self._timestep - self._prev_timestep) / (new_timestamp - self._prev_timestamp)).op.run()
        self._prev_timestamp = new_timestamp
        self._prev_timestep = self._timestep

        # meta state (0-observe, 1-explore, 2-train)
        if self._timestep <= OBSERVE:
            state = 0
        elif self._timestep > OBSERVE and self._timestep <= OBSERVE + EXPLORE:
            state = 1
        else:
            state = 2
        self._meta_state.assign(state).op.run()

        summary = self._session.run(self._summaries)
        self._summarywriter.add_summary(summary, self._timestep)

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
