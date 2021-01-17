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


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers):
        super(Net, self).__init__()
        layers = []
        prev_channels = in_channels
        for channels in hidden_layers:
            layers.append(FCLayer(prev_channels, channels))
            prev_channels = channels
        layers.append(nn.Linear(prev_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DQN:
    def __init__(self, in_channels, out_channels, hidden_layers, lr=0.001, model_load_path=None, **_kwargs):
        self.net = Net(in_channels, out_channels, hidden_layers)
        if model_load_path is not None:
            self.net.load_state_dict(torch.load(model_load_path))
        # self.optimizer = optim.RMSprop(self.net.parameters())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)

    def predict(self, x: np.ndarray) -> torch.Tensor:
        return self.net(torch.Tensor(x).to(self.device))

    def optimize(self, batch, gamma: float):
        states = torch.FloatTensor([transition.state for transition in batch]).to(self.device)
        actions = torch.LongTensor([transition.action for transition in batch]).to(self.device)
        rewards = torch.FloatTensor([transition.reward for transition in batch]).to(self.device)
        next_states = torch.FloatTensor([transition.next_state for transition in batch]).to(self.device)
        dones = torch.BoolTensor([transition.done for transition in batch]).to(self.device)

        # calculate current q-values
        qs = self.predict(states)
        qs = torch.gather(qs, 1, actions.unsqueeze(1))

        # calculate expected q-values using reward and next state
        next_qs = self.predict(next_states)
        target_qs = rewards + gamma * torch.where(dones, torch.tensor(0.), torch.max(next_qs, dim=-1).values)
        target_qs = target_qs.unsqueeze(1)

        loss = F.mse_loss(qs, target_qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, model_save_path: str):
        verify_output_path(model_save_path)
        torch.save(self.net.state_dict(), model_save_path)
