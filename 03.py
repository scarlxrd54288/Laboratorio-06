import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Para guardar y cargar la tabla Q

class SoftmaxQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, tau=1.0, tau_decay=0.999, tau_min=0.01):
        self.env = env 
        self.alpha = alpha #Tasa de aprendizaje
        self.gamma = gamma #Factor de descuento
        self.tau = tau #temperatura
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.q_table = np.zeros((40, 40, self.env.action_space.n))
        self.discrete_os_window = (self.env.observation_space.high - self.env.observation_space.low) / [40, 40]
        self.rewards = []
    
    def get_discrete_state(self, state):
        return tuple(((state - self.env.observation_space.low) / self.discrete_os_window).astype(int))
    
    def choose_action(self, state):
        logits = self.q_table[state]
        exp_values = np.exp(logits / self.tau)
        sum_exp = np.sum(exp_values)
        if sum_exp == 0:
            probabilities = np.ones_like(exp_values) / len(exp_values)  # Distribución uniforme si todos los valores Q son iguales
        else:
            probabilities = exp_values / sum_exp
        action = np.random.choice(self.env.action_space.n, p=probabilities)
        return action
    
    def update_q_table(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state + (action,)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state + (action,)] = new_q
    
    def adjust_tau(self, episode_reward):
        # Ajuste dinámico de tau basado en la recompensa del episodio
        if len(self.rewards) >= 100:
            recent_avg_reward = np.mean(self.rewards[-100:])
            if episode_reward > recent_avg_reward:
                self.tau = max(self.tau * self.tau_decay, self.tau_min)
            else:
                self.tau = min(self.tau / self.tau_decay, 10)  # Aumentar tau para más exploración
        else:
            self.tau = max(self.tau * self.tau_decay, self.tau_min)
    
    def train(self, episodes=1000, render_every=100):
        for episode in range(episodes):
            state = self.get_discrete_state(self.env.reset()[0])
            done = False
            episode_reward = 0
            while not done:
                if episode % render_every == 0:
                    self.env.render()
                action = self.choose_action(state)
                next_state_raw, reward, done, _, _ = self.env.step(action)
                next_state = self.get_discrete_state(next_state_raw)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
            self.rewards.append(episode_reward)
            self.adjust_tau(episode_reward)
            print(f"Episodio {episode + 1}: Recompensa = {episode_reward}, Tau = {self.tau:.4f}")
        print("Entrenamiento completo.")
    
    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards, label='Recompensa por Episodio')
        # Añadir promedio móvil
        window_size = 50
        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.rewards)), moving_avg, label='Promedio Móvil', color='red')
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa Total")
        plt.title("Recompensas por Episodio")
        plt.legend()
        plt.show()
    
    def save_q_table(self, filename="q_table_softmax.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Tabla Q guardada en {filename}.")
    
    def load_q_table(self, filename="q_table_softmax.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Tabla Q cargada desde {filename}.")
        
    def plot_q_table(self, action_index=0):
     #Grafica los valores Q de una acción específica en toda la tabla Q.
    #action_index (int): Índice de la acción que se desea graficar.
        # Verificar que el índice de acción es válido
        if action_index >= self.env.action_space.n:
            print("Índice de acción fuera de rango.")
            return

        # Graficar los valores Q de la acción especificada
        plt.imshow(self.q_table[:, :, action_index], cmap="viridis")
        plt.colorbar()
        plt.title(f"Q-table values for action {action_index}")
        plt.xlabel("State Dimension 1")
        plt.ylabel("State Dimension 2")
        plt.show()

if __name__ == "__main__":
    #env = gym.make("MountainCar-v0")
    env = gym.make("MountainCar-v0", render_mode="human")
    env.metadata['render_fps'] = 200
    agent = SoftmaxQLearningAgent(env)
    # agent.load_q_table()  # Cargar la tabla Q guardada previamente si existe
    agent.train(episodes=5000, render_every=1000)
    env.close()
    agent.plot_rewards()
    agent.save_q_table()  # Guardar la tabla Q después de entrenar

    # Graficar los valores Q de la primera acción
    agent.plot_q_table(action_index=0)  # Cambia el índice para graficar otras acciones si deseas


