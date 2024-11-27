import os
import gymnasium as gym
import numpy as np
import random
import json
from stable_baselines3 import SAC
import torch


def train_manual_sac(env, params, n_steps):
    """
    Entrena un modelo SAC con los parámetros proporcionados y evalúa su desempeño.
    """
    # Extraer los parámetros
    learning_rate = params["learning_rate"]
    buffer_size = params["buffer_size"]
    batch_size = params["batch_size"]
    tau = params["tau"]
    gamma = params["gamma"]
    ent_coef = params["ent_coef"]

    # Crear el modelo SAC
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=1,
    )

    # Entrenar el modelo
    model.learn(total_timesteps=n_steps)

    # Evaluar el modelo
    avg_reward, avg_steps = evaluate_policy(env, model, model, max_total_steps=200, n_episodes=10)
    print(f"Avg Reward: {avg_reward}, Avg Steps: {avg_steps}")

    return model, avg_reward


def evaluate_policy(env, model, policy, max_total_steps=200, n_episodes=5, verbose=False):
    """
    Evalúa la política aprendida por el modelo.
    """
    total_reward = 0
    total_steps = 0
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done and total_steps < max_total_steps:
            selected_action, _ = policy.predict(obs, deterministic=True)  # Usar SAC.predict
            obs, reward, done, truncated, _ = env.step(selected_action)
            episode_reward += reward
            episode_steps += 1
            total_reward += episode_reward
            total_steps += episode_steps

        # Si verbose es True, imprimir los resultados del episodio
        if verbose:
            print(f"Episode {episode + 1} Reward: {episode_reward}, Steps: {episode_steps}")
    
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes
    return avg_reward, avg_steps


if __name__ == "__main__":
    # Configuración del entorno
    env_name = "Ant-v5"
    env = gym.make(env_name)
    seed = 42
    random.seed(seed)
    env.reset(seed=seed)

    # Número de pasos de entrenamiento
    n_steps = 1000

    # Mejores conjuntos de hiperparámetros
    best_params = {
        "learning_rate": 0.001,
        "buffer_size": 1000000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto",
    }
    
    # Configuración del dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Entrenar y guardar los modelos
    print("Training with BEST configuration:")
    model_1, reward_1 = train_manual_sac(env, best_params, n_steps)
    model_1.save("./sac_ant_best")

    # Cerrar el entorno
    env.close()

    # Guardar las recompensas promedio en un archivo JSON
    results = {
        "best_reward": reward_1,
    }
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=4)
