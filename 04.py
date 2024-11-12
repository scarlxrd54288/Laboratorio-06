import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Definir la red de política usando softmax
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax para obtener probabilidades

class SoftmaxGradientAgent:
    def __init__(self, env, gamma=0.99, learning_rate=0.01):
        self.env = env
        self.gamma = gamma
        self.policy_network = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, torch.log(action_probs[0, action])  # Retorna la acción y log-probabilidad

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns)

    def update_policy(self, log_probs, returns):
        # Normalizar los retornos
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calcular la pérdida negativa para maximizar
        policy_loss = -torch.sum(torch.stack(log_probs) * returns)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000):
        rewards_all = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            log_probs = []
            rewards = []
            done = False

            while not done:
                action, log_prob = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

            returns = self.compute_returns(rewards)
            self.update_policy(log_probs, returns)
            total_reward = sum(rewards)
            rewards_all.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Graficar las recompensas
        plt.plot(rewards_all)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Rewards per Episode")
        plt.show()

# Configuración del entorno y ejecución del agente
env = gym.make("MountainCar-v0")
agent = SoftmaxGradientAgent(env)
agent.train(episodes=500)
env.close()
