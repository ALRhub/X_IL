import sys
# sys.path.append("/home/david/2025")
sys.path.append("/hkfs/work/workspace/scratch/ll6323-david_dataset_2/david")

import logging

import cv2
import einops
import numpy as np
import wandb
import robosuite.utils.transform_utils as T
import torch
from omegaconf import ListConfig
from robocasa.utils.env_utils import create_env
from termcolor import cprint
from tqdm import tqdm

from agents.base_agent import BaseAgent
from custom_robocasa.env_wrappers.point_cloud_sampling_wrapper import (
    PointCloudSamplingWrapper,
)
from custom_robocasa.env_wrappers.point_cloud_wrapper import PointCloudWrapper
from custom_robocasa.env_wrappers.segmentation_wrapper import SegmentationWrapper
from custom_robocasa.env_wrappers.segmented_point_cloud_sampling_wrapper import (
    SegmentedPointCloudSamplingWrapper,
)
from custom_robocasa.utils.point_cloud.sampling.fps_pc_sampler import (
    FPSPointCloudSampler,
)
from environments.wrappers.robosuite_wrapper import RobosuiteWrapper
from simulation.base_sim import BaseSim
from utils.hilbert_curve import reorder_point_cloud_with_hilbert_curve

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
        pc_num_points: int = 1024,
        obj_max_num_points: int = 512,
        use_full_point_cloud: bool = False,
        use_segmented_point_cloud: bool = False,
        global_action: bool = False,
        use_pc_color: bool = False,
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode
        self.camera_names = list(camera_names)
        self.global_action = global_action

        self.env_name = env_name

        self.img_height = img_height
        self.img_width = img_width

        self.use_pc_color = use_pc_color
        self.pc_num_points = pc_num_points
        self.obj_max_num_points = obj_max_num_points
        self.use_segmented_point_cloud = use_segmented_point_cloud

        if use_full_point_cloud:
            if use_segmented_point_cloud:
                self.pc_key = "segmented_point_cloud"
            else:
                self.pc_key = "point_cloud"
        else:
            if use_segmented_point_cloud:
                self.pc_key = "segmented_sampled_point_cloud"
            else:
                self.pc_key = "sampled_point_cloud"

        self.pos_key = "robot0_eef_pos" if global_action else "robot0_base_to_eef_pos"
        self.quat_key = (
            "robot0_eef_quat" if global_action else "robot0_base_to_eef_quat"
        )

        # if render:
        #     self.env.reset()
        #     self.env.render()
        #     cv2.destroyAllWindows()

        cprint(f"Using point cloud key: {self.pc_key}", "blue")
        cprint(f"Using position key: {self.pos_key}", "blue")
        cprint(f"Using quaternion key: {self.quat_key}", "blue")

    def test_agent(self, agent: BaseAgent):

        for env in self.env_name:

            print(f"Initializing environment {env}")

            self._init_env(
                env,
                self.img_width,
                self.img_height,
                self.render,
                self.pc_num_points,
                self.obj_max_num_points,
                self.use_segmented_point_cloud
            )

            success_count = 0
            task_completion_hold_count = -1

            for i in range(self.num_episode):
                obs = self.env.reset()

                if self.render:
                    self.env.render()

                agent.reset()

                lang = self.env.get_ep_meta()["lang"]

                for j in tqdm(range(self.max_step_per_episode)):
                    obs_dict = {}
                    obs_dict["lang"] = lang

                    gripper_state = torch.from_numpy(obs["robot0_gripper_qpos"][:1]).float()
                    obs_dict["gripper_state"] = einops.rearrange(
                        gripper_state, "d -> 1 1 d"
                    ).to(self.device)

                    eef_pos = torch.from_numpy(obs[self.pos_key]).float()
                    obs_dict["eef_pos"] = einops.rearrange(eef_pos, "d -> 1 1 d").to(
                        self.device
                    )

                    eef_quat = torch.from_numpy(obs[self.quat_key]).float()
                    obs_dict["eef_quat"] = einops.rearrange(eef_quat, "d -> 1 1 d").to(
                        self.device
                    )

                    sampled_point_cloud = obs[self.pc_key]
                    if not self.use_pc_color:
                        sampled_point_cloud = sampled_point_cloud[:, :3]
                    else:
                        sampled_point_cloud[:, 3:] /= 255.0

                    # sampled_point_cloud = reorder_point_cloud_with_hilbert_curve(np.expand_dims(sampled_point_cloud, axis=0))
                    sampled_point_cloud = np.expand_dims(sampled_point_cloud, axis=0)
                    sampled_point_cloud = torch.from_numpy(sampled_point_cloud).float()

                    obs_dict["point_cloud"] = einops.rearrange(
                        sampled_point_cloud, "1 num_points d -> 1 1 num_points d"
                    ).to(self.device)

                    for cam_name in self.camera_names:

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
            print(f"{env} Success rate: {success_rate}")

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
            pc_num_points,
            obj_max_num_points,
            use_segmented_point_cloud,
    ):
        base_env = create_env(
            env_name=env_name,
            camera_widths=img_width,
            camera_heights=img_height,
            camera_names=self.camera_names,
            has_renderer=render,
            seed=self.seed,
        )

        if use_segmented_point_cloud:
            self.env = SegmentedPointCloudSamplingWrapper(
                PointCloudSamplingWrapper(
                    PointCloudWrapper(
                        SegmentationWrapper(
                            RobosuiteWrapper(base_env),
                            env_name=env_name,
                        ),
                        global_frame=False,
                        get_segmented_pc=True,
                    ),
                    pc_sampler=FPSPointCloudSampler(),
                    num_points=pc_num_points,
                ),
                env_name=env_name,
                obj_sampler=FPSPointCloudSampler(),
                rest_sampler=FPSPointCloudSampler(),
                num_points=pc_num_points,
                obj_max_num_points=obj_max_num_points,
            )
        else:
            self.env = PointCloudSamplingWrapper(
                PointCloudWrapper(RobosuiteWrapper(base_env), global_frame=False),
                pc_sampler=FPSPointCloudSampler(),
                num_points=pc_num_points,
            )