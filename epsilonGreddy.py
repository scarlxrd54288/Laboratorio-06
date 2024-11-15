# Importamos las bibliotecas necesarias
import gymnasium as gym  # Librería para crear y gestionar entornos de entrenamiento
import numpy as np  # Librería para trabajar con arreglos y operaciones numéricas
import matplotlib.pyplot as plt  # Librería para generar gráficos


# Definimos el número de divisiones en cada dimensión del estado (posición y velocidad)
num_bins = (18, 18)  # Número de divisiones para discretizar el espacio de estados, Vector
bins = [
    np.linspace(-1.2, 0.6, num_bins[0] - 1),  # Divisiones para la posición del coche
    np.linspace(-0.07, 0.07, num_bins[1] - 1)  # Divisiones para la velocidad del coche
]

# Función para discretizar un estado continuo
# Convertimos el estado continuo en un estado discretizado usando los "bins"
#digitize: encontrar el índice de un valor 
def discretize_state(state):
    return tuple(np.digitize(s, bins[i]) for i, s in enumerate(state))

# Función para entrenar el agente en el entorno
def train(episodes):
    env = gym.make('MountainCar-v0')  # Creamos el entorno de MountainCar

    # Inicializamos la tabla Q para la política Epsilon-Greedy
    # new_q será una matriz con las dimensiones de los bins del estado y las acciones posibles
    new_q = np.zeros(num_bins + (env.action_space.n,))

    # Parámetros del algoritmo
    alpha = 0.1  # Tasa de aprendizaje: cuánto se ajusta la tabla Q en cada paso
    gamma = 0.99  # Factor de descuento: influencia de las recompensas futuras 
    epsilon_greedy = 0.5  # Probabilidad de explorar (en vez de explotar la mejor acción)
    epsilon_decay_rate = 0.995  # Tasa de decaimiento de epsilon después de cada episodio
    rng = np.random.default_rng()  # Generador de números aleatorios para la exploración

    # Inicializamos los arrays para almacenar las recompensas y pasos por episodio, zeros:array de ceros
    rewards_per_episode_epsilon_greedy = np.zeros(episodes)  # Recompensas acumuladas por episodio
    steps_per_episode_epsilon_greedy = np.zeros(episodes, dtype=int)  # Pasos tomados en cada episodio
    actions_per_episode = []  # Lista de acciones tomadas en cada episodio

    # Comienza el ciclo de entrenamiento
    for i in range(episodes):
        # Reiniciamos el entorno al comienzo de cada episodio
        state, _ = env.reset()
        state = discretize_state(state)  # Discretizamos el estado inicial

        # Variables de control del episodio
        terminated = False  # Bandera que indica si el episodio terminó por alcanzar la meta (cuando gana)
        truncated = False  # Bandera que indica si el episodio terminó por exceder el límite de pasos (cuando pierde)
        total_reward_epsilon_greedy = 0  # Recompensa acumulada en este episodio
        steps_epsilon_greedy = 0  # Contador de pasos en este episodio
        actions_in_episode = []  # Lista de acciones tomadas en este episodio

        # Comienza la simulación del episodio usando Epsilon-Greedy, mientras no gana ni pierde
        while not terminated and not truncated:
            # Selección de acción utilizando la política Epsilon-Greedy, si es mayor explora 
            if rng.random() < epsilon_greedy:
                # Exploramos (acción aleatoria), cuando explora toma una accion existente
                action = env.action_space.sample()
            else:
                # Explotamos (tomamos la mejor acción según la tabla Q), argmax: indice del valor maximo
                action = np.argmax(new_q[state])

            # Ejecutamos la acción en el entorno y obtenemos el siguiente estado, recompensa, si el episodio ha terminado
            new_state, reward, terminated, truncated, _ = env.step(action) #<-Ejecutamos la acción en el entorno
            new_state = discretize_state(new_state)  # Discretizamos el nuevo estado

            # Guardamos la acción tomada en este episodio
            actions_in_episode.append(action)

            # Ajustamos la recompensa: 1 si alcanzamos la meta, -1 si no
            if terminated:
                reward = 1  # Recompensa por alcanzar la meta
            else:
                reward = -1  # Penalización por cada paso sin alcanzar la meta

            total_reward_epsilon_greedy += reward  # Acumulamos la recompensa total del episodio

            # Actualizamos la tabla Q , con la formula  
            new_q[state + (action,)] = new_q[state + (action,)] + alpha * (reward + gamma * (np.max(new_q[new_state]) - new_q[state + (action,)]))
            
            state = new_state  # Actualizamos el "estado" actual 
            steps_epsilon_greedy += 1  # Aumentamos el contador de pasos

            #numero maximo de pasos limite es 200, no puede ser mayor pero si puede ser menos
            # Si el número de pasos supera 200, truncamos el episodio
            if steps_epsilon_greedy >= 200:
                truncated = True

        # Decaemos el valor de epsilon para reducir la exploración a medida que el agente aprende
        epsilon_greedy = max(epsilon_greedy - epsilon_decay_rate, 0)
        
        # Almacenamos la recompensa y los pasos del episodio
        rewards_per_episode_epsilon_greedy[i] = total_reward_epsilon_greedy
        steps_per_episode_epsilon_greedy[i] = steps_epsilon_greedy
        actions_per_episode.append(actions_in_episode)

        # Imprimimos el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio {i + 1} - Epsilon-Greedy: Recompensa {rewards_per_episode_epsilon_greedy[i]}, Pasos {steps_per_episode_epsilon_greedy[i]}")

    env.close()  # Cerramos el entorno después de entrenar

    # Encontramos el mejor episodio (el que tomó menos pasos)
    best_episode = np.argmin(steps_per_episode_epsilon_greedy)
    best_episode_actions = actions_per_episode[best_episode]
    print(f"Mejor episodio con {steps_per_episode_epsilon_greedy[best_episode]} pasos")
    
    # Imprime la tabla Q en la terminal
    print("\nTabla Q después del entrenamiento:\n")
    print(new_q) 

    # Exportamos la tabla Q y las acciones del mejor episodio a un archivo .txt
    with open("Q_table.txt", "w") as f:
        f.write("Tabla Q Epsilon-Greedy:\n")
        f.write(np.array2string(new_q, separator=', '))  # Guardamos la tabla Q
        f.write(f"\n\nMejor episodio: {best_episode + 1} con {steps_per_episode_epsilon_greedy[best_episode]} pasos\n")
        f.write("Acciones del mejor episodio:\n")
        f.write(", ".join(map(str, best_episode_actions)))  # Guardamos las acciones del mejor episodio

    # Calculamos la recompensa media por cada 100 episodios
    avg_rewards_epsilon_greedy = np.zeros(episodes // 100)
    for t in range(episodes // 100):
        avg_rewards_epsilon_greedy[t] = np.mean(rewards_per_episode_epsilon_greedy[t * 100:(t + 1) * 100])

    # Graficamos la recompensa media por cada 100 episodios
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards_epsilon_greedy, label='Epsilon-Greedy - Recompensa Media')
    plt.xlabel('Bloques de 100 episodios')
    plt.ylabel('Recompensa media')
    plt.legend()
    plt.title('Rendimiento de la política Epsilon-Greedy')
    plt.show()
    
    

# Entrenamos el agente durante 10,000 episodios
if __name__ == "__main__":
    train(10000)
