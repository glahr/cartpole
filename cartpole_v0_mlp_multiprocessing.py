import random
import gym
import numpy as np
from collections import deque
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
MAX_EPOCHS = 10

observation_space = 4
action_space = 2



class DQNSolver:

    def __init__(self, observation_space, action_space, q_remember, q_net_weights):

        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam
        import keras.backend as K
        import tensorflow as tf

        config = K.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = K.tf.Session(config=config)
        K.set_session(session)

        # K.set_session(tf.Session())

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
        # print("aqui")
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

            # self.counter += 1
            # if self.counter%1000 == 0:
            #     print("counter_train = ", self.counter)


        if self.q_net_weights.empty():
            self.q_net_weights.put(self.model.get_weights())


class DQNActions:

    def __init__(self, action_space, q_net_weights):

        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam
        import keras.backend as K
        import tensorflow as tf
        config = K.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = K.tf.Session(config=config)
        K.set_session(session)
        # K.set_session(tf.Session())

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

        self.action_space = action_space
        self.q_net_weights = q_net_weights

    def act_mlp(self, state, exploration_rate):
        if not self.q_net_weights.empty():
            self.model.set_weights(self.q_net_weights.get())

        if np.random.rand() < exploration_rate:
            return random.randrange(self.action_space)
            
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # def remember(self, state, action, reward, next_state, done):

def train_net(observation_space, action_space, q_remember, q_net_weights):
    dqn_solver = DQNSolver(observation_space, action_space, q_remember, q_net_weights)
    while True:
        # print("AQUI")
        dqn_solver.experience_replay()

def cartpole(q_remember, q_net_weights, flag_finished):
    env = gym.make(ENV_NAME)
    # observation_space = env.observation_space.shape[0]
    print("observation_space = ", observation_space)
    # action_space = env.action_space.n
    print("action_space = ", action_space)
    run = 0

    exploration_rate = EXPLORATION_MAX
    counter = 0

    dqn_actions = DQNActions(action_space, q_net_weights)

    # while True:
    while(run < MAX_EPOCHS):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0

        while True:
            # print("SIMULATION")
            # time.sleep(0.005)
            step += 1
            action = dqn_actions.act_mlp(state, exploration_rate)

            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            q_remember.put([state, action, reward, state_next, terminal, exploration_rate])
            # print("STATE-----")
            state = state_next

            # counter += 1
            # if counter%100 == 0:
            #     print("counter_sim = ", counter)

            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(exploration_rate) + ", score: " + str(step))
                break

        exploration_rate *= EXPLORATION_DECAY
        exploration_rate = max(EXPLORATION_MIN, exploration_rate)

    if flag_finished.empty():
        flag_finished.put(True)

if __name__ == "__main__":

    # multiprocessing
    q_remember = Queue(maxsize = 1)
    flag_finished = Queue(maxsize = 1)
    q_net_weights = Queue(maxsize = 1)
    finished = False
    process_simulation = Process(target=cartpole, args=(q_remember, q_net_weights, flag_finished))
    process_train_net = Process(target=train_net, args=(observation_space, action_space, q_remember, q_net_weights))
    # process_simulation.start()
    process_train_net.start()
    cartpole(q_remember, q_net_weights, flag_finished)

    # process_train_mlp.join()
    # process_simulation.join()
    # counter = 0
    while process_simulation.is_alive() and not finished:
        if not flag_finished.empty():
            finished = flag_finished.get()
        # print(q_remember.empty())
        # print(counter)
        # counter += 1

    # process_simulation.terminate()
    process_train_net.terminate()
