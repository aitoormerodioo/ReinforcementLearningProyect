import os
import gymnasium as gym
import numpy as np
import random
import json
from stable_baselines3 import SAC
import torch


def train_manual_sac(env, model, n_steps):
    """
    Entrena un modelo SAC por el número de pasos especificados y evalúa su desempeño.
    """
    # Entrenar el modelo
    model.learn(total_timesteps=n_steps)

    # Evaluar el modelo
    avg_reward, avg_steps = evaluate_policy(env, model, model, max_total_steps=200, n_episodes=10)
    print(f"Avg Reward: {avg_reward}, Avg Steps: {avg_steps}")

    return model, avg_reward


def evaluate_policy(env, model, policy, max_total_steps=5000, n_episodes=5, verbose=False):
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
    n_steps = 500_000

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

    # Ruta donde se guarda el modelo
    model_path = "sac_ant_best_model/sac_ant_model.zip"

    # Comprobar si existe un modelo guardado
    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Loading the model...")
        model = SAC.load(model_path, env=env)
    else:
        print("No model found. Creating a new model...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=best_params["learning_rate"],
            buffer_size=best_params["buffer_size"],
            batch_size=best_params["batch_size"],
            tau=best_params["tau"],
            gamma=best_params["gamma"],
            ent_coef=best_params["ent_coef"],
            verbose=1,
        )

    # Entrenar el modelo (ya sea cargado o nuevo)
    print("Training the model...")
    model, avg_reward = train_manual_sac(env, model, n_steps)
    
    # Guardar el modelo actualizado
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Cerrar el entorno
    env.close()

    # Guardar las recompensas promedio en un archivo JSON
    results = {
        "final_reward": avg_reward,
    }
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=4)
