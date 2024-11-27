import gymnasium
import numpy as np

env = gymnasium.make(
    'Ant-v5',
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
    max_episode_steps=1000,
    render_mode='human',
)

# Inicializar el entorno
obs, info = env.reset(seed=42)

# Ejecutar episodios aleatorios
for step in range(200_000):  # Cambia el número de pasos si lo necesitas
    action = env.action_space.sample()  # Generar acción aleatoria
    obs, reward, terminated, truncated, info = env.step(action)  # Ejecutar acción en el entorno
    
    # Mostrar el reward en cada paso
    print(f"Step {step + 1}: Reward = {reward}")
    
    if terminated or truncated:  # Reiniciar si el episodio termina
        print("Episodio terminado. Reiniciando...")
        obs, info = env.reset()

# Cerrar el entorno
env.close()
