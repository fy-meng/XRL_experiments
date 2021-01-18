from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd

from utils import ReplayBuffer, verify_output_path
from model import DQN, QRDQN

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
        return self.net.predict(state).detach().cpu().numpy()  # shape = (b, m, c)

    def optimize(self):
        batch: List[Transition] = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        self.net.optimize(batch, self.gamma)

    def save_model(self, model_save_path: str):
        self.net.save_model(model_save_path)


class QRDQNCropAgent(Agent):
    WATER_VALUES = np.array([0, 2, 4, 6, 8])
    NITROGEN_VALUES = np.array([0, 125, 250, 375, 500])
    PHOSPHORUS_VALUES = np.array([0, 50, 100, 150, 200])
    HARVEST_VALUES = np.array([0, 1])

    def __init__(self, state_size, _num_actions, batch_size=64, gamma=0.999, epsilon=0.9,
                 epsilon_decay=0.99995, buffer_size=None, **kwargs):
        num_actions = len(self.WATER_VALUES) * len(self.NITROGEN_VALUES) \
                      * len(self.PHOSPHORUS_VALUES) * len(self.HARVEST_VALUES)
        super(QRDQNCropAgent, self).__init__(state_size, num_actions, buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.net = QRDQN(state_size, num_actions, **kwargs)

    def get_action(self, state: np.ndarray):
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = np.argmax(np.mean(self.get_q_values(state), axis=1), axis=-1)

        self.epsilon *= self.epsilon_decay

        # convert action index to actual action values
        water_idx = action_idx % len(self.WATER_VALUES)
        nitrogen_idx = (action_idx // len(self.WATER_VALUES)) % len(self.NITROGEN_VALUES)
        phosphorus_idx = (action_idx // len(self.WATER_VALUES) // len(self.NITROGEN_VALUES)) \
                         % len(self.PHOSPHORUS_VALUES)
        harvest_idx = (action_idx // len(self.WATER_VALUES) // len(self.NITROGEN_VALUES)
                       // len(self.PHOSPHORUS_VALUES)) % len(self.HARVEST_VALUES)

        action = np.array([
            self.WATER_VALUES[water_idx],
            self.NITROGEN_VALUES[nitrogen_idx],
            self.PHOSPHORUS_VALUES[phosphorus_idx],
            self.HARVEST_VALUES[harvest_idx]
        ])

        return action
    
    def store_transition(self, transition: Transition):
        new_transition = Transition(
            episode=transition.episode,
            state=transition.state,
            action=self.action_to_idx(transition.action),
            reward=transition.reward,
            next_state=transition.next_state,
            done=transition.done,
            q_values=transition.q_values
        )
        super(QRDQNCropAgent, self).store_transition(new_transition)

    @staticmethod
    def action_to_idx(action):
        water, nitrogen, phosphorus, harvest = action
        water_idx = np.nonzero(QRDQNCropAgent.WATER_VALUES == water)[0][0]
        nitrogen_idx = np.nonzero(QRDQNCropAgent.NITROGEN_VALUES == nitrogen)[0][0]
        phosphorus_idx = np.nonzero(QRDQNCropAgent.PHOSPHORUS_VALUES == phosphorus)[0][0]
        harvest_idx = np.nonzero(QRDQNCropAgent.HARVEST_VALUES == harvest)[0][0]

        idx = harvest_idx
        idx = idx * len(QRDQNCropAgent.PHOSPHORUS_VALUES) + phosphorus_idx
        idx = idx * len(QRDQNCropAgent.NITROGEN_VALUES) + nitrogen_idx
        idx = idx * len(QRDQNCropAgent.WATER_VALUES) + water_idx

        return idx

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.net.predict(state).detach().cpu().numpy()

    def optimize(self):
        batch: List[Transition] = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        self.net.optimize(batch, self.gamma)

    def save_model(self, model_save_path: str):
        self.net.save_model(model_save_path)
