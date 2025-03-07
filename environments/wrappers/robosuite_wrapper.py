import gym
import numpy as np
from robosuite.utils.transform_utils import (
    axisangle2quat,
    quat2axisangle,
    quat2mat,
    euler2mat,
    mat2quat,
)


class RobosuiteWrapper(gym.Wrapper):
    """
    This wrapper does the following changes to the action and observation:
    - Flips the image and segmentation observations
    - Converts the depth observation from depth image to meters
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        obs_dict = self.process_observation(obs_dict)

        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()

        obs_dict = self.process_observation(obs_dict)

        return obs_dict

    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)

        obs_dict = self.process_observation(obs_dict)

        return obs_dict

    def process_observation(self, obs_dict):
        for key in obs_dict:
            if "image" in key or "segmentation" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
            elif "depth" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
                obs_dict[key] = self.depthimg2Meters(obs_dict[key])

        return obs_dict

    def _check_success(self):
        return self.env._check_success()

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image
