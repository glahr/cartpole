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

        self.model_copy = None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, q_remember):
        print("entrei aqui")
        if not q_remember.empty():
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

            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        else:
            print("fila vazia, mano!")

    def copy_keras_model(self):
        self.model_copy = keras.models.clone_model(self.model)
        self.model_copy.set_weights(self.model.get_weights())
        return self.model_copy

class DQNActions:

    def __init__(self, exploration_rate, action_space, model):
        self.exploration_rate = exploration_rate
        self.action_space = action_space
        self.model = model

    def act_mlp(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # q_values = sess.run(self.fc, feed_dict={self.X: state.reshape((1,1,4))})
        # q_values = self.model.predict(state)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # def remember(self, state, action, reward, next_state, done):


def cartpole():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    print("observation_space = ", observation_space)
    action_space = env.action_space.n
    print("action_space = ", action_space)
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0

    # multiprocessing
    q_remember = Queue()
    process_train_mlp = Process(target=dqn_solver.experience_replay, args=(q_remember,))

    process_train_mlp.start()


    while True:
    # while(run < MAX_EPOCHS):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        # dqn_solver.copy_keras_model()
        dqn_actions = DQNActions(dqn_solver.exploration_rate, action_space, dqn_solver.copy_keras_model())

        while True:
            step += 1
            #env.render()

            action = dqn_actions.act_mlp(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break

        # dqn_solver.experience_replay()

        dqn_solver.exploration_rate *= EXPLORATION_DECAY
        dqn_solver.exploration_rate = max(EXPLORATION_MIN, dqn_solver.exploration_rate)

if __name__ == "__main__":
    cartpole()
