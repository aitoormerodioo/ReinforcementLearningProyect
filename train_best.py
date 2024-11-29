import os
import gymnasium as gym
import numpy as np
import random
import json
from stable_baselines3 import SAC
import torch
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy


def train_manual_sac(env, model, n_steps):
    """
    Entrena un modelo SAC por el número de pasos especificados y evalúa su desempeño.
    """
    # Entrenar el modelo
    model.learn(total_timesteps=n_steps, log_interval=1)

    # Evaluar el modelo
    avg_reward, avg_steps = evaluate_policy(env, model, model, max_steps_per_episode=5_000, n_episodes=1, verbose=True)
    print(f"Avg Reward: {avg_reward}, Avg Steps: {avg_steps}")

    return model, avg_reward


def evaluate_policy(env, model, policy, max_steps_per_episode, n_episodes=1, verbose=False):
    """
    Evalúa la política aprendida por el modelo con un límite máximo de pasos por episodio.
    """
    total_reward = 0
    total_steps = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done and episode_steps < max_steps_per_episode:
            selected_action, _ = policy.predict(obs, deterministic=True)  # Usar SAC.predict
            obs, reward, done, truncated, _ = env.step(selected_action)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        total_reward += episode_reward

        if verbose:
            print(f"Episode {episode + 1} Reward: {episode_reward}, Steps: {episode_steps}, Done: {done}, Truncated: {truncated}")
    
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes

    return avg_reward, avg_steps


if __name__ == "__main__":
    # Configuración del entorno
    env_name = "Ant-v5"
    
    # Uso de rutas relativas
    current_dir = os.path.dirname(os.path.realpath(__file__))  # Directorio actual donde se ejecuta el script
    xml_path = os.path.join(current_dir, 'mujoco_menagerie', 'unitree_go1', 'scene.xml')  # Ruta relativa

    env = gym.make('Ant-v5', 
                   xml_file=xml_path,
                   forward_reward_weight=1,
                   ctrl_cost_weight=0.1,
                   contact_cost_weight=0.5,
                   healthy_reward=0.1,
                   main_body=1,
                   healthy_z_range=(0.2, 1),
                   reset_noise_scale=0,
                   frame_skip=1,
                   max_episode_steps=1_000)
    
    seed = 42
    random.seed(seed)
    env.reset(seed=seed)

    # Número de pasos de entrenamiento
    n_steps = 300_000

    # Mejores conjuntos de hiperparámetros ENCONTRADOS
    best_params = {
        "learning_rate": 0.0003,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.99,
        "buffer_size": 1_000_000,
        "learning_starts": 1000,
        "train_freq": (1, "episode"),
        "gradient_steps": 1,
        "target_update_interval": 1,
    }

    # Rutas relativas para modelos y logs
    model_dir = os.path.join(current_dir, 'sac_ant_best_model')
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    
    training_r = os.path.join(current_dir, 'training_results')
    os.makedirs(training_r, exist_ok=True)

    # Obtener la fecha actual para nombres únicos
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Buscar el modelo más reciente en el directorio
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if model_files:
        # Encontrar el modelo más reciente basado en el nombre
        model_files.sort(reverse=True)
        latest_model_path = os.path.join(model_dir, model_files[0])
        print(f"Found existing model: {latest_model_path}. Continuing training from this model.")
        model = SAC.load(latest_model_path, env=env)
    else:
        print("No existing model found. Creating a new model...")
        model = SAC(
            "MlpPolicy",
            env,
            tensorboard_log=log_dir,
            learning_rate=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            tau=best_params["tau"],
            gamma=best_params["gamma"],
            buffer_size=best_params["buffer_size"],
            learning_starts=best_params["learning_starts"],
            train_freq=best_params["train_freq"],
            gradient_steps=best_params["gradient_steps"],
            target_update_interval=best_params["target_update_interval"],
            verbose=2,
        )

    # Entrenar el modelo
    print("Training the model...")
    model, avg_reward = train_manual_sac(env, model, n_steps)
    
    # Guardar el modelo actualizado con la fecha actual
    model_path = os.path.join(model_dir, f"sac_ant_model_{date_str}.zip")
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Guardar las recompensas promedio en un archivo JSON
    results = {
        "model_name": f"sac_ant_model_{date_str}.zip",
        "final_reward": avg_reward,
    }
    results_path = os.path.join(current_dir, f"training_results/training_results_{date_str}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved at {results_path}")
