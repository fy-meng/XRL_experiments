import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import verify_output_path


class FCLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.relu(self.fc(x), inplace=True)


class FCNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers):
        super(FCNet, self).__init__()
        layers = []
        prev_channels = in_channels
        for channels in hidden_layers:
            layers.append(FCLayer(prev_channels, channels))
            prev_channels = channels
        layers.append(nn.Linear(prev_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NeuralNet:
    def predict(self, x: np.ndarray) -> torch.Tensor:
        raise NotImplemented

    def optimize(self, batch, gamma: float):
        raise NotImplemented

    def save_model(self, model_save_path: str):
        raise NotImplemented


class DQN(NeuralNet):
    def __init__(self, in_channels, out_channels, hidden_layers, lr=0.001, model_load_path=None, **_kwargs):
        self.net = FCNet(in_channels, out_channels, hidden_layers)
        if model_load_path is not None:
            self.net.load_state_dict(torch.load(model_load_path))
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)

    def predict(self, x: np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).to(self.device)
        return self.net(x)

    def optimize(self, batch, gamma: float):
        states = torch.FloatTensor([transition.state for transition in batch]).to(self.device)
        actions = torch.LongTensor([transition.action for transition in batch]).to(self.device)
        rewards = torch.FloatTensor([transition.reward for transition in batch]).to(self.device)
        next_states = torch.FloatTensor([transition.next_state for transition in batch]).to(self.device)
        dones = torch.BoolTensor([transition.done for transition in batch]).to(self.device)

        # calculate current q-values
        expected_qs = self.predict(states)
        expected_qs = expected_qs.gather(1, actions.unsqueeze(1))

        # calculate expected q-values using reward and next state
        next_qs = self.predict(next_states)
        target_qs = rewards + gamma * torch.where(dones,
                                                  torch.tensor(0.).to(self.device),
                                                  torch.max(next_qs, dim=-1).values)
        target_qs = target_qs.unsqueeze(1)

        loss = F.mse_loss(expected_qs, target_qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, model_save_path: str):
        verify_output_path(model_save_path)
        torch.save(self.net.state_dict(), model_save_path)


class QRDQN(NeuralNet):
    def __init__(self, in_channels, out_channels, hidden_layers, num_bins=16, lr=0.001, kappa=1.0,
                 model_load_path=None, **_kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bins = num_bins

        self.net = FCNet(in_channels, out_channels * num_bins, hidden_layers).to(self.device)
        if model_load_path is not None:
            self.net.load_state_dict(torch.load(model_load_path))

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.kappa = kappa
        self.tau = torch.FloatTensor([i / self.num_bins for i in range(1, self.num_bins + 1)]).to(self.device)

    def predict(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).to(self.device)
        return self.net(x).reshape(-1, self.out_channels, self.num_bins)

    def optimize(self, batch, gamma: float):
        batch_size = len(batch)

        states = torch.FloatTensor([transition.state for transition in batch]).to(self.device)
        actions = torch.LongTensor([transition.action for transition in batch]).to(self.device)
        rewards = torch.FloatTensor([transition.reward for transition in batch]).to(self.device)
        next_states = torch.FloatTensor([transition.next_state for transition in batch]).to(self.device)
        dones = torch.BoolTensor([transition.done for transition in batch]).to(self.device)

        # calculate current q-values
        expected_qs = self.predict(states)  # shape = (b, m, c)
        actions = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.num_bins)  # shape = (b, 1, c)
        expected_qs = expected_qs.gather(1, actions)  # shape = (b, 1, c)
        expected_qs = expected_qs.transpose(1, 2)  # shape = (b, c, 1)
        assert expected_qs.shape == (batch_size, self.num_bins, 1)

        # calculate expected q-values using reward and next state
        next_qs = self.predict(next_states)  # shape = (b, n, m)
        best_next_actions = torch.argmax(torch.mean(next_qs, dim=2), dim=1, keepdim=True)  # shape = (b, 1)
        best_next_actions = best_next_actions.unsqueeze(-1).expand(-1, 1, self.num_bins)  # shape = (b, 1, c)
        target_qs = next_qs.gather(1, best_next_actions)  # shape = (b, 1, c)
        dones = dones.unsqueeze(-1).unsqueeze(-1)  # shape = (b, 1, 1)
        rewards = rewards.unsqueeze(-1).unsqueeze(-1)  # shape = (b, 1, 1)
        target_qs = rewards + gamma * torch.where(dones,
                                                  torch.tensor(0.).to(self.device),
                                                  target_qs)  # shape = (b, 1, c)
        assert target_qs.shape == (batch_size, 1, self.num_bins)

        td_errors = target_qs - expected_qs
        assert td_errors.shape == (batch_size, self.num_bins, self.num_bins)

        huber_loss = self.huber_loss(td_errors)
        quantile_loss = torch.abs(self.tau - (td_errors.detach() < 0).float()) * huber_loss / 1.0

        loss = quantile_loss.sum(dim=1).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def huber_loss(self, td_errors):
        loss = torch.where(td_errors.abs() <= self.kappa,
                           0.5 * td_errors.pow(2),
                           self.kappa * (td_errors.abs() - 0.5 * self.kappa))
        assert loss.shape == (td_errors.shape[0], self.num_bins, self.num_bins)
        return loss

    def save_model(self, model_save_path: str):
        verify_output_path(model_save_path)
        torch.save(self.net.state_dict(), model_save_path)
