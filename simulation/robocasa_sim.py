import logging

import cv2
import einops
import numpy as np
import robosuite.utils.transform_utils as T
import torch
import wandb
from omegaconf import ListConfig
from robocasa.utils.env_utils import create_env
from tqdm import tqdm

from agents.base_agent import BaseAgent
from environments.wrappers.robosuite_wrapper import RobosuiteWrapper
from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)


class RoboCasaSim(BaseSim):
    def __init__(
        self,
        env_name: str,
        camera_names: ListConfig[str],
        img_height: int,
        img_width: int,
        num_episode: int,
        max_step_per_episode: int,
        seed: int,
        device: str,
        render: bool = True,
        n_cores: int = 1,
        if_vision: bool = False,
        global_action: bool = False,
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode
        self.camera_names = list(camera_names)
        self.global_action = global_action

        self.env_name = env_name

        self.img_height = img_height
        self.img_width = img_width

    def test_agent(self, agent: BaseAgent):

        for env in self.env_name:

            print(f"Initializing environment {env}")

            self._init_env(
                env,
                self.img_width,
                self.img_height,
                self.render,
            )

            success_count = 0
            task_completion_hold_count = -1

            for i in range(self.num_episode):
                obs = self.env.reset()

                if self.render:
                    self.env.render()

                agent.reset()

                lang = self.env.get_ep_meta()["lang"]

                print(f"episode {i}: ", lang)

                for j in tqdm(range(self.max_step_per_episode)):
                    obs_dict = {}
                    obs_dict["lang"] = lang

                    gripper_state = torch.from_numpy(obs["robot0_gripper_qpos"]).float()
                    gripper_state = einops.rearrange(gripper_state, "d -> 1 1 d").to(
                        self.device
                    )

                    joint_pos = torch.from_numpy(obs["robot0_joint_pos_cos"]).float()
                    joint_pos = einops.rearrange(joint_pos, "d -> 1 1 d").to(self.device)

                    robot_state = torch.cat([gripper_state, joint_pos], dim=-1)
                    obs_dict["robot_states"] = robot_state

                    for cam_name in self.camera_names:

                        # cv2.imshow(f"{cam_name}_image", obs[f"{cam_name}_image"])
                        # cv2.waitKey(1)

                        rgb = (
                            torch.from_numpy(obs[f"{cam_name}_image"].copy())
                            .float()
                            .permute(2, 0, 1)
                            / 255.0
                        )
                        rgb = einops.rearrange(rgb, "c h w -> 1 1 c h w").to(self.device)
                        obs_dict[f"{cam_name}_image"] = rgb

                    action = agent.predict(obs_dict).cpu().numpy()

                    action = np.concatenate(
                        [action, np.array([0, 0, 0, 0, -1])]
                    )

                    # action = np.concatenate(
                    #     [action[1:], action[:1], np.array([0, 0, 0, 0, -1])]
                    # )
                    if self.global_action:
                        action = self.get_local_action(action)

                    obs, _, done, _ = self.env.step(action)

                    if self.render:
                        self.env.render()

                    if self.env._check_success():
                        if task_completion_hold_count > 0:
                            task_completion_hold_count -= (
                                1  # latched state, decrement count
                            )
                        else:
                            task_completion_hold_count = (
                                10  # reset count on first success timestep
                            )
                    else:
                        task_completion_hold_count = (
                            -1
                        )  # null the counter if there's no success

                    if task_completion_hold_count == 0:
                        success_count += 1
                        done = True

                    if done:
                        break

            success_rate = success_count / self.num_episode
            print(f"Success rate: {success_rate}")

            wandb.log({f"{env}_average_success": success_rate})

            self.env.close()

    def _get_local_action(self, global_action: np.ndarray) -> np.ndarray:
        base_mat = self.env.sim.data.get_site_xmat(
            f"mobilebase{self.env.robots[0].idn}_center"
        )

        global_action_pos = global_action[:3]
        global_action_axis_angle = global_action[3:6]
        global_action_mat = T.quat2mat(T.axisangle2quat(global_action_axis_angle))

        local_action_pos = base_mat.T @ global_action_pos
        local_action_mat = base_mat.T @ global_action_mat @ base_mat
        local_action_axis_angle = T.quat2axisangle(T.mat2quat(local_action_mat))

        local_action = np.concatenate(
            [local_action_pos, local_action_axis_angle, global_action[6:]]
        )
        return local_action

    def _init_env(
        self,
        env_name,
        img_width,
        img_height,
        render,
    ):
        base_env = create_env(
            env_name=env_name,
            camera_widths=img_width,
            camera_heights=img_height,
            camera_names=self.camera_names,
            has_renderer=render,
            seed=self.seed,
        )

        self.env = RobosuiteWrapper(base_env)
