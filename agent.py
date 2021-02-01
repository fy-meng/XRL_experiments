from collections import namedtuple
from typing import List

import numpy as np

from utils import ReplayBuffer, verify_output_path
from model import DQN, QRDQN, CADQN
from sarfa_saliency import computeSaliencyUsingSarfa

Transition = namedtuple('Transition',
                        ('episode', 'state', 'action', 'reward', 'next_state', 'done', 'q_values'))


class Agent:
    def __init__(self, state_size, num_actions, mode, buffer_size=None, **_kwargs):
        assert mode in ('train', 'test')
        self.mode = mode
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
        history = {field: np.array([
            transition.__getattribute__(field) for transition in self.buffer.data
        ]) for field in Transition._fields}
        np.savez(path, **history)


class DQNAgent(Agent):
    def __init__(self, state_size, num_actions, batch_size=64, gamma=0.999, epsilon=0.9, epsilon_decay=0.99995,
                 **kwargs):
        super(DQNAgent, self).__init__(state_size, num_actions, **kwargs)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.net = DQN(state_size, num_actions, **kwargs)

    def get_action(self, state: np.ndarray):
        if self.mode == 'train' and np.random.random() < self.epsilon:
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


class CropAgent(Agent):
    WATER_VALUES = np.array([0, 2, 4, 6, 8])
    NITROGEN_VALUES = np.array([0, 0.5, 1, 1.5, 2])
    PHOSPHORUS_VALUES = np.array([0, 0.5, 1, 1.5, 2])

    SALIENCY_TRIALS = 10
    PERTURB_COEFF = 0.1

    def __init__(self, state_size, num_actions, compute_saliency, **kwargs):
        super(CropAgent, self).__init__(state_size, num_actions, **kwargs)

        self.compute_saliency = compute_saliency
        self.saliencies = []
        self.state_min = None  # min of each feature seen so far, used for perturbation
        self.state_max = None  # max of each feature seen so far, used for perturbation

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
        super(CropAgent, self).store_transition(new_transition)

        if self.compute_saliency:
            self.saliencies.append(self.get_saliency(transition.state, transition.q_values))

    def get_saliency(self, state: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        raise NotImplemented

    @staticmethod
    def action_to_idx(action: np.ndarray) -> int:
        # water, nitrogen, phosphorus, harvest = action
        water, nitrogen, phosphorus = action
        water_idx = np.nonzero(QRDQNCropAgent.WATER_VALUES == water)[0][0]
        nitrogen_idx = np.nonzero(QRDQNCropAgent.NITROGEN_VALUES == nitrogen)[0][0]
        phosphorus_idx = np.nonzero(QRDQNCropAgent.PHOSPHORUS_VALUES == phosphorus)[0][0]

        idx = phosphorus_idx
        idx = idx * len(QRDQNCropAgent.NITROGEN_VALUES) + nitrogen_idx
        idx = idx * len(QRDQNCropAgent.WATER_VALUES) + water_idx

        return idx

    @staticmethod
    def idx_to_action(idx: int) -> np.ndarray:
        water_idx = idx % len(QRDQNCropAgent.WATER_VALUES)
        nitrogen_idx = (idx // len(QRDQNCropAgent.WATER_VALUES)) % len(QRDQNCropAgent.NITROGEN_VALUES)
        phosphorus_idx = (idx // len(QRDQNCropAgent.WATER_VALUES) // len(QRDQNCropAgent.NITROGEN_VALUES)) \
                         % len(QRDQNCropAgent.PHOSPHORUS_VALUES)

        action = np.array([
            CropAgent.WATER_VALUES[water_idx],
            CropAgent.NITROGEN_VALUES[nitrogen_idx],
            CropAgent.PHOSPHORUS_VALUES[phosphorus_idx],
        ])

        return action

    def update_state_value_range(self, state: np.ndarray):
        if self.state_min is None:
            self.state_min = state.copy()
            self.state_max = state.copy()
        else:
            self.state_min = np.minimum(self.state_min, state)
            self.state_max = np.maximum(self.state_max, state)

    def perturb(self, state: np.ndarray, idx: int) -> np.ndarray:
        amount = np.random.random() * self.PERTURB_COEFF * (self.state_max[idx] - self.state_min[idx])
        perturbed_state = state.copy()
        perturbed_state[idx] += amount
        return perturbed_state

    def save_history(self, path):
        verify_output_path(path)
        history = {field: np.array([
            transition.__getattribute__(field) for transition in self.buffer.data
        ]) for field in Transition._fields}
        history['saliency'] = np.array(self.saliencies)
        np.savez(path, **history)


class QRDQNCropAgent(CropAgent):
    def __init__(self, state_size, _num_actions, batch_size=64, gamma=0.999, epsilon=0.9,
                 epsilon_decay=0.99995, **kwargs):
        num_actions = len(self.WATER_VALUES) * len(self.NITROGEN_VALUES) \
                      * len(self.PHOSPHORUS_VALUES)
        super(QRDQNCropAgent, self).__init__(state_size, num_actions, **kwargs)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.net = QRDQN(state_size, self.num_actions, **kwargs)

    def get_action(self, state: np.ndarray):
        if self.mode == 'train' and np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = self.get_q_values(state).mean(axis=1).argmax(axis=-1)

        self.epsilon *= self.epsilon_decay

        # convert action index to actual action values
        action = self.idx_to_action(action_idx)

        return action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.net.predict(state).detach().cpu().numpy()

    def get_saliency(self, state: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        assert state.size == self.state_size, "saliency cannot be computed during training"

        self.update_state_value_range(state)

        saliency = np.zeros_like(state)
        action: int = q_values.mean(axis=1).argmax()
        q_values_dict = {i: q for i, q in enumerate(q_values.mean(axis=1).squeeze())}

        for _ in range(self.SALIENCY_TRIALS):
            for i in range(self.state_size):
                perturbed_state = self.perturb(state, i)
                perturbed_q_values = self.get_q_values(perturbed_state)
                perturbed_q_values_dict = {j: q for j, q in enumerate(perturbed_q_values.mean(axis=1).squeeze())}

                saliency[i] += computeSaliencyUsingSarfa(action,
                                                         q_values_dict,
                                                         perturbed_q_values_dict)[0] / self.SALIENCY_TRIALS

        return saliency

    def optimize(self):
        batch: List[Transition] = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        self.net.optimize(batch, self.gamma)

    def save_model(self, model_save_path: str):
        self.net.save_model(model_save_path)


class DQNCropAgent(CropAgent):
    def __init__(self, state_size, _num_actions, batch_size=64, gamma=0.999, epsilon=0.9,
                 epsilon_decay=0.99995, **kwargs):
        num_actions = len(self.WATER_VALUES) * len(self.NITROGEN_VALUES) \
                      * len(self.PHOSPHORUS_VALUES)
        super(DQNCropAgent, self).__init__(state_size, num_actions, **kwargs)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.net = DQN(state_size, self.num_actions, **kwargs)

    def get_action(self, state: np.ndarray):
        if self.mode == 'train' and np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = self.get_q_values(state).argmax(axis=-1)

        self.epsilon *= self.epsilon_decay

        # convert action index to actual action values
        action = self.idx_to_action(action_idx)

        return action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.net.predict(state).detach().cpu().numpy()

    def get_saliency(self, state: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        assert state.size == self.state_size, "saliency cannot be computed during training"

        self.update_state_value_range(state)

        saliency = np.zeros_like(state)
        action: int = q_values.argmax()
        q_values_dict = {i: q / 100 for i, q in enumerate(q_values.squeeze())}

        for _ in range(self.SALIENCY_TRIALS):
            for i in range(self.state_size):
                perturbed_state = self.perturb(state, i)
                perturbed_q_values = self.get_q_values(perturbed_state)
                perturbed_q_values_dict = {j: q / 100 for j, q in enumerate(perturbed_q_values.squeeze())}

                saliency[i] += computeSaliencyUsingSarfa(action,
                                                         q_values_dict,
                                                         perturbed_q_values_dict)[0] / self.SALIENCY_TRIALS

        return saliency

    def optimize(self):
        batch: List[Transition] = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        self.net.optimize(batch, self.gamma)

    def save_model(self, model_save_path: str):
        self.net.save_model(model_save_path)


class CADQNAgent(Agent):
    def __init__(self, state_size, num_actions, action_min, action_max, batch_size=64, gamma=0.999, epsilon=0.9,
                 epsilon_decay=0.99995, **kwargs):
        super(CADQNAgent, self).__init__(state_size, num_actions, **kwargs)

        self.state_size = state_size
        self.num_actions = num_actions
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.net = CADQN(state_size, self.num_actions, **kwargs)

    def get_action(self, state: np.ndarray):
        if self.mode == 'train' and np.random.random() < self.epsilon:
            action = self.action_min + np.random.rand(self.num_actions) * (self.action_max - self.action_min)
        else:
            action = self.net.predict(state).detach().cpu().numpy()[:self.num_actions]

        self.epsilon *= self.epsilon_decay

        return action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.net.predict(state).detach().cpu().numpy()[-1]

    def get_saliency(self, state: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def optimize(self):
        batch: List[Transition] = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        self.net.optimize(batch, self.gamma)

    def save_model(self, model_save_path: str):
        self.net.save_model(model_save_path)

