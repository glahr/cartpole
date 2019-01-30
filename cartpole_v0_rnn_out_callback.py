import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam, Adadelta
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# from scores.score_logger import ScoreLogger

class MetricsCallbacksClass(keras.callbacks.Callback):
    def __init__(self):
        self.loss = []

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        # self.loss = logs.get('loss')
        # print(logs.get('loss'))
        self.loss.append(logs.get('loss'))
        return

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999
MAX_EPOCHS = 40
# sess = tf.Session()


# # RNN
# timesteps = 1
# num_input = 4
# X = tf.placeholder("float", [None, timesteps, num_input], name="X_plc")
# Y = tf.placeholder("float", [None, 2], name="Y_plc")
# n_h1 = 20
# n_h2 = 5
# n_cells_layers = [n_h2]
# lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_) for n_ in n_cells_layers]
# multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
# outputs, states = tf.nn.dynamic_rnn(multi_lstm_cell, inputs=X, dtype=tf.float32, scope = "dynamic_rnn")
# fc = tf.contrib.layers.fully_connected(outputs[:,-1], 2, activation_fn = None, scope="my_fully_connected")
# # self.prediction = tf.nn.softmax()
# # with tf.name_scope("metrics"):
# xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc, labels=Y, name="xentropy")
# optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
# loss_op = tf.reduce_mean(xentropy, name="loss_op")
# tf.summary.scalar("loss", loss_op)
# train_op = optimizer.minimize(loss_op, name = "train_op")

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(LSTM(4, input_shape = (1, observation_space), return_sequences = True))
        self.model.add(LSTM(2))
        self.model.add(Dense(action_space, activation = "relu"))
        adadelta = Adadelta(lr=LEARNING_RATE, rho=0.95, epsilon=1e-6, decay=0.95)
        self.model.compile(loss="mse", optimizer=adadelta)

        # self.model = Sequential()
        # self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        # self.model.add(Dense(24, activation="relu"))
        # self.model.add(Dense(self.action_space, activation="linear"))
        # self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

        self.history = MetricsCallbacksClass()


        # # RNN
        # timesteps = 1
        # num_input = observation_space
        # self.X = tf.placeholder("float", [None, timesteps, num_input], name="X_plc")
        # self.Y = tf.placeholder("float", [None, action_space], name="Y_plc")
        # n_h1 = 20
        # n_h2 = 5
        # n_cells_layers = [n_h2]
        # lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_) for n_ in n_cells_layers]
        # multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        # outputs, states = tf.nn.dynamic_rnn(multi_lstm_cell, inputs=self.X, dtype=tf.float32, scope = "dynamic_rnn")
        # self.fc = tf.contrib.layers.fully_connected(outputs[:,-1], action_space, activation_fn = None, scope="my_fully_connected")
        # # self.prediction = tf.nn.softmax()
        # # with tf.name_scope("metrics"):
        # xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc, labels=self.Y, name="xentropy")
        # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        # self.loss_op = tf.reduce_mean(xentropy, name="loss_op")
        # tf.summary.scalar("loss", self.loss_op)

            # prediction = tf.nn.softmax(self.fc, name = "prediction")
            # correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1), name = "correct_pred")
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")

        # self.train_op = optimizer.minimize(self.loss_op, name = "train_op")
        # tf.summary.scalar('accuracy', accuracy)
        # tf.summary.scalar('loss_op', self.loss_op)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act_mlp(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # q_values = sess.run(self.fc, feed_dict={self.X: state.reshape((1,1,4))})
        state.shape = (1,1,4)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def act_rnn(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # q_values = sess.run(fc, feed_dict={X: state.reshape((1,1,4))})
        state.shape = (1,1,4)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            state_next.shape = (1,1,4)
            state.shape = (1,1,4)
            # print(state_next)
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update


            self.model.fit(state, q_values, verbose=0, callbacks = [self.history])

            #RNN train
            # print("state_reshape = ",state.reshape((1,1,4)))
            # print("q_values = ", q_values)
            # sess.run(train_op, feed_dict = {X: state.reshape((1,1,4)), Y: q_values})

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)




def cartpole():
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    print("observation_space = ", observation_space)
    action_space = env.action_space.n
    print("action_space = ", action_space)
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0

    # with sess:
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('tensorboard/3')
    # # writer.add_graph(sess.graph)
    # init = tf.global_variables_initializer()
    # sess.run(init)


    while True:
    # while(run < MAX_EPOCHS):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()

            # if step % 2 == 0:
            #     summary = sess.run(merged, feed_dict = {X: state.reshape((1,1,4)), Y: [[None, action]]})
            #     writer.add_summary(summary, step)
                # writer.flush()

            action = dqn_solver.act_mlp(state)
            action_rnn = dqn_solver.act_rnn(state)
            # action = action_rnn
            # print("mlp = ", action)
            # print("rnn = ", action_rnn)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                if run % 15 == 0:
                    plt.figure()
                    plt.plot(dqn_solver.history.loss)
                    plt.show()
                    if run % 46 == 0:
                        dqn_solver.history.loss = []
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()
