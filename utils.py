from collections import deque
from configparser import ConfigParser
import os
import random
from typing import Dict

import gym
import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def format_runtime(runtime, precision=3):
    runtime = round(runtime, precision)

    hours = int(runtime / 3600)
    minutes = int((runtime - hours * 3600) / 60)
    seconds = int(runtime - hours * 3600 - minutes * 60)
    milliseconds = int(round(runtime - int(runtime), 3) * 1000)

    result = ''
    if hours >= 1:
        result += f'{hours}h '
    if minutes >= 1:
        result += f'{minutes}m '
    if seconds >= 1:
        result += f'{seconds}s '
    if milliseconds > 0:
        result += f'{milliseconds}ms'

    return result


def verify_output_path(output_path):
    output_dir = os.path.split(output_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_config() -> Dict[str, object]:
    parser = ConfigParser()
    parser.read('./config.ini')
    config = {}
    for section in parser.sections():
        for key, item in parser[section].items():
            # convert to list of int
            if key in ('hidden_layers',):
                config[key] = [int(s) for s in item.split(',')]
                continue
            # try convert to int
            try:
                config[key] = int(item)
                continue
            except ValueError:
                pass
            # convert to float
            try:
                config[key] = float(item)
                continue
            except ValueError:
                pass
            # convert to bool
            if item == 'True' or item == 'False':
                config[key] = bool(item)
                continue
            # check for empty value
            if item == '':
                config[key] = None
                continue
            # otherwise, kept as str
            else:
                config[key] = item
    return config


def get_state_size(env: gym.Env) -> int:
    if isinstance(env.observation_space, gym.spaces.box.Box):
        state_size = 1
        for i in range(len(env.observation_space.shape)):
            state_size *= env.observation_space.shape[i]
        return state_size
    elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        return env.observation_space.n
    else:
        raise ValueError('Observation Space type is not supported')


def get_num_actions(env: gym.Env) -> int:
    if isinstance(env.action_space, gym.spaces.box.Box):
        num_actions = 1
        for i in range(len(env.action_space.shape)):
            num_actions *= env.action_space.shape[i]
        return num_actions
    elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
        return env.action_space.n
    else:
        raise ValueError('Observation Space type is not supported')


class ReplayBuffer:
    def __init__(self, max_size=None):
        self.data = deque(maxlen=max_size)

    def append(self, x):
        self.data.append(x)

    def __len__(self):
        return len(self.data)

    def sample(self, n):
        if len(self.data) < n:
            return None
        else:
            idx = np.random.choice(np.arange(len(self.data)), n)
            return [self.data[i] for i in idx]
