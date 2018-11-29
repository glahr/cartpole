import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import time

from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999
MAX_EPOCHS = 40


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))


        # RNN
        timesteps = 1
        num_input = observation_space
        self.X = tf.placeholder("float", [None, timesteps, num_input], name="X_plc") #STUDY
        self.Y = tf.placeholder("float", [None, action_space], name="Y_plc")
        n_h1 = 20
        n_h2 = 2
        n_cells_layers = [n_h2]
        lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_) for n_ in n_cells_layers]
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        outputs, states = tf.nn.dynamic_rnn(multi_lstm_cell, inputs=self.X, dtype=tf.float32, scope = "dynamic_rnn")
        self.fc = tf.contrib.layers.fully_connected(outputs[:,-1], action_space, activation_fn = None, scope="my_fully_connected")
        # self.prediction = tf.nn.softmax()
        # with tf.name_scope("metrics"):
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc, labels=self.Y, name="xentropy")
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        loss_op = tf.reduce_mean(xentropy, name="loss_op")
        self.train_op = optimizer.minimize(loss_op, name = "train_op")
        tf.summary.scalar("loss", loss_op)

            # prediction = tf.nn.softmax(self.fc, name = "prediction")
            # correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1), name = "correct_pred")
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")
        # tf.summary.scalar('accuracy', accuracy)
        # tf.summary.scalar('loss_op', self.loss_op)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act_mlp(self, state):
        #check if a random number is smaller than exploration_rate. If not, returns the prediction by the function approximator
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)

        q_values = self.model.predict(state)
        return np.argmax(q_values[0]) #it just takes q_values[0] because np.shape(q_values) = (1,3) and np.shape(q_values[0]) = (3,)

    def act_rnn(self, state, sess):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)

        q_values = sess.run(self.fc, feed_dict={self.X: state.reshape((1,1,4))})
        return np.argmax(q_values[0])

    def experience_replay(self, sess):
        #if the memory has not at least BATCH_SIZE itens, we cannot train our network, due to the fact that we cannot sample that much. However, we are going to reach for the first batch when len(self.memory) = BATCH_SIZE + 1, i.e., we are going to sample experiences very close to each other.
        if len(self.memory) < BATCH_SIZE:
            return
        #when properly populated, we get a sample from the memory, contaning: state, action, reward, state_next, terminal
        batch = random.sample(self.memory, BATCH_SIZE)
        #now, we are going to update our q_values
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                #when I run model.predict(next_state), I'm outputting the actions, that's why we take the greedy. Here, I'm calculating the right side of the Bellman equation for Q-learning and storing it at the auxiliar variable q_update
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            #now, I'm taking the predicted q_value for the current state
            q_values = self.model.predict(state)
            #as the situation is from the past, we can classify if it was action 0 or 1, and then, store the the new Q-value.
            q_values[0][action] = q_update

            #And by storing, we mean, training with this new set our model: here is supervised learning
            self.model.fit(state, q_values, verbose=0)

            #RNN train
            sess.run(self.train_op, feed_dict = {self.X: state.reshape((1,1,4)), self.Y: q_values})

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)




def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    print("observation_space = ", observation_space)
    action_space = env.action_space.n
    print("action_space = ", action_space)
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0

    # with sess:
    # sess = tf.Session()

    # print(sess.run(tf.report_uninitialized_variables()))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('tensorboard/3')
        writer.add_graph(sess.graph)
        while True:
        # while(run < MAX_EPOCHS):
            run += 1 #new episode
            state = env.reset()  # resetting the environment so it doesn't start with previous states
            state = np.reshape(state, [1, observation_space]) #reshaping for a row vector
            step = 0 #tries in one episode

            # if run % 2 == 0 and run > BATCH_SIZE:
            #     summary = sess.run(merged, feed_dict = {dqn_solver.X: state.reshape((1,1,4)), dqn_solver.Y: sess.run(dqn_solver.fc, feed_dict={dqn_solver.X: state.reshape((1,1,4))})})
            #     writer.add_summary(summary, run)
            #     writer.flush()

            while True:
                step += 1
                #env.render()

                action = dqn_solver.act_mlp(state) #take an action based on the estimation of the dqn at state
                # action_rnn = dqn_solver.act_rnn(state, sess) #mine
                # action = action_rnn
                state_next, reward, terminal, info = env.step(action) #return of the env.step call
                reward = reward if not terminal else -reward #if it is over, he gives a negative reward. otherewise, it just keeps increasing
                state_next = np.reshape(state_next, [1, observation_space]) #reshape of the received state
                dqn_solver.remember(state, action, reward, state_next, terminal) #update of the experience memory with the most recent collection
                state = state_next #update of the next state
                if terminal: #if it is over, he plots the final condition and breaks the inner true loop
                    print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                    # score_logger.add_score(step, run) #this is his score logger, it is not needed
                    break
                dqn_solver.experience_replay(sess) #at the end of the step, he runs the experience replay to train


if __name__ == "__main__":
    cartpole()
