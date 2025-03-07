import gym
from utils.point_cloud.pc_generator import PointCloudGenerator

class PointCloudWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cam_names = env.camera_names
        self.pc_generator = PointCloudGenerator(env.sim, self.cam_names, env.camera_widths[0], env.camera_heights[0])

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        return obs_dict, reward, done, info
    
    def reset(self):
        obs_dict = self.env.reset()
        self.pc_generator.sim = self.env.sim
        
        obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        return obs_dict
    
    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)
        self.pc_generator.sim = self.env.sim
        
        obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        return obs_dict
    
    def get_point_cloud(self, obs_dict):
        imgs = {cam: obs_dict[f"{cam}_image"] for cam in self.cam_names}
        depths = {cam: obs_dict[f"{cam}_depth"] for cam in self.cam_names}
        
        return self.pc_generator.get_point_cloud(imgs, depths)
    
    def _check_success(self):
        return self.env._check_success()

