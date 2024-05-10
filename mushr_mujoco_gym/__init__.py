from gymnasium.envs.registration import register

register(
    id='MushrBlock-v0',
    entry_point='mushr_mujoco_gym.envs:MushrBlockEnv',
    max_episode_steps=1000,
)
