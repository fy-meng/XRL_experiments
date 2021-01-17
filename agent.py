from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd

from utils import ReplayBuffer, verify_output_path
from model import DQN

Transition = namedtuple('Transition',
                        ('episode', 'state', 'action', 'reward', 'next_state', 'done', 'q_values'))


class Agent:
    def __init__(self, state_size, num_actions, buffer_size=None):
        self.state_size = state_size
        self.num_actions = num_actions

        self.buffer = ReplayBuffer(buffer_size)

    def get_action(self, state: np.ndarray):
        raise NotImplemented

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def optimize(self):
        raise NotImplemented

    def save_model(self, model_save_path: str):
        raise NotImplemented

    def store_transition(self, transition):
        self.buffer.append(transition)

    def save_history(self, path):
        verify_output_path(path)
        pd.to_pickle(self.buffer.data, path)


class DQNAgent(Agent):
    def __init__(self, state_size, num_actions, batch_size=64, gamma=0.999, epsilon=0.9,
                 epsilon_decay=0.99995, buffer_size=None, **kwargs):
        super(DQNAgent, self).__init__(state_size, num_actions, buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.net = DQN(state_size, num_actions, **kwargs)

    def get_action(self, state: np.ndarray):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.get_q_values(state), axis=-1)
        self.epsilon *= self.epsilon_decay
        return action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.net.predict(state).detach().cpu().numpy()

    def optimize(self):
        batch: List[Transition] = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        self.net.optimize(batch, self.gamma)

    def save_model(self, model_save_path: str):
        self.net.save_model(model_save_path)
