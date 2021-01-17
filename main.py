import time

import gym

from agent import *
from utils import format_runtime, set_random_seed, load_config, get_state_size, get_num_actions


def train(env: gym.Env, agent: Agent, num_episodes: int, model_save_path='./trained_models/dqn.pth',
          save_history=False, train_history_save_path='./output/history_train.pkl', verbose=True, **_kwargs):
    for i in range(num_episodes):
        t = time.time()

        state = env.reset()
        done = False
        total_return = 0
        steps = 0
        while not done:
            q_values = agent.get_q_values(state)
            action = agent.get_action(state)
            next_state, reward, done, _info = env.step(action)
            total_return += reward

            agent.store_transition(Transition(
                episode=i,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state if not done else np.zeros_like(next_state),
                done=done,
                q_values=q_values
            ))
            agent.optimize()

            state = next_state
            steps += 1

        runtime = time.time() - t
        if verbose:
            len_trail_num = len(str(num_episodes))
            print(f'episode {i:0{len_trail_num}}: '
                  f'steps = {steps}, '
                  f'return = {total_return:.2f}, '
                  f'time = {format_runtime(runtime)}')

    agent.save_model(model_save_path)

    if save_history:
        agent.save_history(train_history_save_path)


def test(env: gym.Env, agent: Agent, num_episodes: int, save_history=False,
         test_history_save_path='./output/history_test.pkl', verbose=True, **_kwargs):
    for i in range(num_episodes):
        t = time.time()

        state = env.reset()
        done = False
        total_return = 0
        steps = 0
        while not done:
            q_values = agent.get_q_values(state)
            action = q_values.argmax()
            next_state, reward, done, _info = env.step(action)
            total_return += reward

            agent.store_transition(Transition(
                episode=i,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state if not done else np.zeros_like(next_state),
                done=done,
                q_values=q_values
            ))

            state = next_state
            steps += 1

        runtime = time.time() - t
        if verbose:
            len_trail_num = len(str(num_episodes))
            print(f'episode {i:0{len_trail_num}}: '
                  f'steps = {steps}, '
                  f'return = {total_return:.2f}, '
                  f'time = {format_runtime(runtime)}')

    if save_history:
        agent.save_history(test_history_save_path)


def main():
    config = load_config()

    assert config['mode'] in ('train', 'test')
    func = train if config['mode'] == 'train' else test

    assert config['env_name'] in ('LunarLander-v2',)
    env = gym.make(config['env_name'])
    state_size = get_state_size(env)
    num_actions = get_num_actions(env)

    if config['seed'] is not None:
        set_random_seed(config['seed'])

    assert config['agent_type'] in ('DQNAgent',)
    agent = globals()[str(config['agent_type'])](state_size, num_actions, **config)

    func(env, agent, **config)


if __name__ == '__main__':
    main()
