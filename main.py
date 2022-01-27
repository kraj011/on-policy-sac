# TO SET EVERYTHING UP:
# - Run pypi install instructions here: https://github.com/google/brax

# The plan is:
# - Copy over the spinning up SAC code
# - Modify it so we can run it from this file
# - Add in the Brax stuff using the GymWrapper so it works with our code (https://github.com/google/brax/blob/main/brax/envs/wrappers.py)
# - Make sure basic SAC works with this setup
# - Then add in the on policy SAC code (more steps will be added once this step is reached)
import src.sac as sac
import gym
import functools
from brax import envs
from brax.envs import to_torch
import torch

ENV_NAME = "halfcheetah"
MINI_BATCH_SIZE = 16


def create_env():
    entry_point = functools.partial(envs.create_gym_env, env_name='ant')
    if 'brax-ant-v0' not in gym.envs.registry.env_specs:
        gym.register('brax-ant-v0', entry_point=entry_point)

    # create a gym environment that contains 16 parallel ant environments
    gym_env = gym.make("brax-ant-v0", batch_size=MINI_BATCH_SIZE)

    # wrap it to interoperate with torch data structures
    gym_env = to_torch.JaxToTorchWrapper(gym_env)
    return gym_env


if __name__ == "__main__":

    gym_env = create_env()

    # jit compile env.reset
    obs = gym_env.reset()

    # jit compile env.step
    action = torch.rand(gym_env.action_space.shape) * 2 - 1
    obs, reward, done, info = gym_env.step(action)

    print(obs, reward, done, info)
