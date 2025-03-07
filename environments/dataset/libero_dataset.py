import logging
import random
import pickle

import cv2
import h5py
import os
import torch
import numpy as np
from environments.dataset.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)


class LiberoDataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 32,
            action_dim: int = 7,
            state_dim: int = 45,
            max_len_data: int = 136,
            window_size: int = 1,
            start_idx: int = 0,
            traj_per_task: int = 1,
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size
        )

        self.data_dir = sim_framework_path(self.data_directory)
        logging.info("The dataset is loading from {}".format(self.data_dir))  # show the dataset directory

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.data_directory = data_directory

        task_suite = os.path.basename(data_directory)
        task_emb_dir = sim_framework_path("task_embeddings")

        with open(task_emb_dir + "/" + task_suite + ".pkl", 'rb') as f:
            tasks = pickle.load(f)

        data_embs = []
        actions = []
        masks = []
        agentview_rgb = []
        eye_in_hand_rgb = []

        all_states = []

        file_list = os.listdir(self.data_dir)

        for file in file_list:
            if not file.endswith('.hdf5'):
                continue

            filename = os.path.basename(file).split('.')[0][:-5]
            task_emb = tasks[filename]

            f = h5py.File(os.path.join(self.data_dir, file), 'r')

            log.info("Loading demo: {}".format(file))

            demo_keys_list = list(f["data"].keys())

            indices = np.argsort([int(elem[5:]) for elem in demo_keys_list])

            # load the states and actions in demos according to demo_keys_list
            for i in indices[start_idx: start_idx + traj_per_task]:

                demo_name = demo_keys_list[i]
                demo = f["data"][demo_name]
                demo_length = demo.attrs["num_samples"]

                # zero_states = np.zeros((1, self.max_len_data, self.state_dim), dtype=np.float32)
                zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                # zero_rewards = np.zeros((1, self.max_len_data), dtype=np.float32)
                # zero_dones = np.zeros((1, self.max_len_data), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                # states_data = demo['states'][:]
                action_data = demo['actions'][:]
                # rewards_data = demo['rewards'][:]
                # dones_data = demo['dones'][:]

                # zero_states[0, :demo_length, :] = states_data  # would be T0, ...,Tn-1, Tn, 0, 0
                zero_actions[0, :demo_length, :] = action_data
                # zero_rewards[0, :demo_length] = rewards_data
                # zero_dones[0, :demo_length] = dones_data
                zero_mask[0, :demo_length] = 1

                # the_last_state = states_data[-1][:]
                the_last_action = action_data[-1][:]
                # the_last_reward = rewards_data[-1]
                # the_last_done = dones_data[-1]

                # zero_agentview = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                # zero_inhand = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                agent_view = demo['obs']['agentview_rgb'][:]
                eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

                joint_states = demo['obs']['joint_states'][:]
                gripper_states = demo['obs']['gripper_states'][:]

                robot_states = np.concatenate((joint_states, gripper_states), axis=-1)

                # test_img = agent_view[0]
                # test_img = test_img[::-1, :, :]
                # test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
                # cv2.imshow("test_img", test_img)
                # cv2.waitKey(0)

                # states.append(zero_states)
                actions.append(zero_actions)
                # rewards.append(zero_rewards)
                # dones.append(zero_dones)
                masks.append(zero_mask)

                agentview_rgb.append(agent_view)
                eye_in_hand_rgb.append(eye_in_hand)

                all_states.append(robot_states)

                data_embs.append(task_emb)

            f.close()

        # self.states = torch.from_numpy(np.concatenate(states)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()  # shape: B, T, D

        self.agentview_rgb = agentview_rgb
        self.eye_in_hand_rgb = eye_in_hand_rgb

        self.all_states = all_states

        self.data_embs = data_embs
        self.tasks = tasks

        # self.rewards = torch.from_numpy(np.concatenate(rewards)).to(device).float()
        # self.dones = torch.from_numpy(np.concatenate(dones)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.agentview_rgb)

        self.slices = self.get_slices()

    def get_slices(self):  #Extract sample slices that meet certain conditions
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.agentview_rgb[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        obs = {}

        task_emb = self.data_embs[i]

        agentview_rgb = self.agentview_rgb[i][start:start+1]
        eye_in_hand_rgb = self.eye_in_hand_rgb[i][start:start+1]

        robot_states = self.all_states[i][start:start+1]

        task_emb = task_emb.to(self.device).float()

        agentview_rgb = torch.from_numpy(agentview_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        obs["agentview_image"] = agentview_rgb
        obs["eye_in_hand_image"] = eye_in_hand_rgb
        obs["lang_emb"] = task_emb

        obs["robot_states"] = torch.from_numpy(robot_states).to(self.device).float()

        return obs, act, mask
