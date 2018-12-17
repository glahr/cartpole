import random
import gym
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import time
from multiprocessing import Process, Queue

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 20000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
MAX_EPOCHS = 1500

observation_space = 4
action_space = 2



class DQNSolver:

    def __init__(self, observation_space, action_space, q_remember, q_net_weights):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.q_remember = q_remember
        self.q_net_weights = q_net_weights

        self.model_copy = None
        self.counter = 0


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        # print("TRAINING")

        if not self.q_remember.empty():
            aux = q_remember.get()
            state = aux[0]
            action = aux[1]
            reward = aux[2]
            next_state = aux[3]
            done = aux[4]
            self.exploration_rate = aux[5]
            self.remember(state, action, reward, next_state, done)

            self.counter += 1
            if self.counter%1000 == 0:
                print("counter = ", self.counter)

        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0)

        # print("original = ", self.model.get_weights())

        # self.exploration_rate *= EXPLORATION_DECAY
        # self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    # def copy_keras_model(self):
            # self.model_copy = keras.models.clone_model(self.model)
            # self.model_copy.set_weights(self.model.get_weights())

            self.q_net_weights.put(self.model.get_weights())
        # return self.model_copy

    # def run(self):


class DQNActions:

    def __init__(self, exploration_rate, action_space, model, q_net_weights):
        self.exploration_rate = exploration_rate
        self.action_space = action_space
        self.model = keras.models.clone_model(model)
        self.q_net_weights = q_net_weights
        # if not q_net_weights.empty():
        #     self.model.set_weights(q_net_weights.get())

        # print("copied = ", self.model.get_weights())

    def act_mlp(self, state):
        if not self.q_net_weights.empty():
            self.model.set_weights(self.q_net_weights.get())

        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # q_values = sess.run(self.fc, feed_dict={self.X: state.reshape((1,1,4))})
        # q_values = self.model.predict(state)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # def remember(self, state, action, reward, next_state, done):

def experience_replay(observation_space, action_space, q_remember, q_net_weights):
    dqn_solver = DQNSolver(observation_space, action_space, q_remember, q_net_weights)
    while True:
        # print("AQUI")
        dqn_solver.experience_replay()

def cartpole(q_remember, q_net_weights):
    env = gym.make(ENV_NAME)
    # observation_space = env.observation_space.shape[0]
    print("observation_space = ", observation_space)
    # action_space = env.action_space.n
    print("action_space = ", action_space)
    # dqn_solver = DQNSolver(observation_space, action_space, q_remember, q_net_weights)
    run = 0

    model = Sequential()
    model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_space, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    exploration_rate = EXPLORATION_MAX

    dqn_actions = DQNActions(exploration_rate, action_space, model, q_net_weights)

    # while True:
    while(run < MAX_EPOCHS):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0

        while True:
            # print("SIMULATION")
            time.sleep(0.05)
            step += 1
            action = dqn_actions.act_mlp(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            q_remember.put([state, action, reward, state_next, terminal, exploration_rate])
            # print("STATE-----")
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(exploration_rate) + ", score: " + str(step))
                break

        exploration_rate *= EXPLORATION_DECAY
        exploration_rate = max(EXPLORATION_MIN, exploration_rate)

if __name__ == "__main__":

    # multiprocessing
    q_remember = Queue(maxsize = 2)
    q_net_weights = Queue(maxsize = 1)
    process_simulation = Process(target=cartpole, args=(q_remember, q_net_weights))
    process_train_mlp = Process(target=experience_replay, args=(observation_space, action_space, q_remember, q_net_weights))
    process_simulation.start()
    process_train_mlp.start()

    # process_train_mlp.join()
    # process_simulation.join()
