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
import brax
