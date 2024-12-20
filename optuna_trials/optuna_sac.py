import os
import gymnasium as gym
import numpy as np
import random
import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import SAC
import json

def evaluate_policy(env, model, policy, max_total_steps=200, n_episodes=5, verbose=False):
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
        #if verbose:
            #print(f"Episode {episode + 1} Reward: {episode_reward}, Steps: {episode_steps}")
    
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes
    return avg_reward, avg_steps

# Función objetivo para optimización con Optuna
def objective(trial):
    # Hiperparámetros para SAC, optimizados por Optuna
    algo_name = trial.suggest_categorical("algo_name", ["SAC"])

    # Parámetros específicos de SAC
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.80, 0.995)
    buffer_size = trial.suggest_int("buffer_size", 10_000, 1_000_000, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 512)
    tau = trial.suggest_float("tau", 0.005, 0.02)
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.1, 0.2, 0.3])

    if algo_name == "SAC":
        model = SAC("MlpPolicy", env, learning_rate=lr, gamma=gamma, buffer_size=buffer_size, 
                    batch_size=batch_size, tau=tau, ent_coef=ent_coef, verbose=0)
    
    # Entrenar el modelo
    model.learn(total_timesteps=200_000)

    # Evaluar la política aprendida
    avg_reward, avg_steps = evaluate_policy(env, model, model.policy, n_episodes=10, verbose=True)

    return avg_reward

if __name__ == "__main__":
    # Crear el entorno de MuJoCo (Ant-v5)
    env = gym.make('Ant-v5', 
                   xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
                   forward_reward_weight=0,
                   ctrl_cost_weight=0,
                   contact_cost_weight=0,
                   healthy_reward=0,
                   main_body=1,
                   healthy_z_range=(0, np.inf),
                   include_cfrc_ext_in_observation=True,
                   exclude_current_positions_from_observation=False,
                   reset_noise_scale=0,
                   frame_skip=1,
                   max_episode_steps=1_000_000)

    # Fijar la semilla para reproducibilidad
    seed = 42
    random.seed(seed)
    env.reset(seed=seed)

    # Eliminar logs anteriores y crear el directorio de Optuna si no existe
    os.system("rm -rf ./logs/")
    # IF WE WANT TO RESTART
    os.system("rm -rf ./optuna/")
    os.system("rm -rf ./optuna.db")
    os.system("mkdir -p ./optuna/")

    # Configuración del estudio de Optuna
    storage_file = f"sqlite:///optuna/optuna.db"
    study_name = "ant_v5_sac"
    full_study_dir_path = f"optuna/{study_name}"
    tpe_sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=tpe_sampler, direction='maximize', 
                                study_name=study_name, storage=storage_file, load_if_exists=True)

    n_trials = 20  # Número de pruebas de optimización

    # Iniciar la optimización
    print(f"Searching best hyperparameter in {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    # Cerrar el entorno
    env.close()

    # Obtener el mejor trial y guardarlo
    best_trials = sorted(study.get_trials(), key=lambda t: t.value, reverse=True)[:3]

    # Prepare data to save for the three best trials
    best_trials_params = {}
    for i, best_trial in enumerate(best_trials):
        trial_data = {
            "value": best_trial.value, 
            "params": best_trial.params, 
            "number": best_trial.number  
        }
        best_trials_params[f"trial_{i + 1}"] = trial_data

        # Save the data in a JSON file
        os.system(f"mkdir -p {full_study_dir_path}")
        with open(f"{full_study_dir_path}/best_trials.json", "w") as best_trials_file:
            json.dump(best_trials_params, best_trials_file, indent=4)

    # Visualizar los resultados y guardar los gráficos
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{full_study_dir_path}/optimization_history.html")
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(f"{full_study_dir_path}/contour.html")
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{full_study_dir_path}/slice.html")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{full_study_dir_path}/param_importances.html")
