import json
import os

import h5py
import torch
from termcolor import cprint

from environments.dataset.base_dataset import TrajectoryDataset


class RobocasaDataset(TrajectoryDataset):
    def __init__(
        self,
        cam_names: list[str],
        env_name: list[str],
        data_directory: os.PathLike,
        device: str = "cpu",
        obs_dim: int = 20,
        action_dim: int = 7,
        max_len_data: int = 256,
        window_size: int = 1,
        global_action: bool = False,
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        self.env_name = env_name
        self.cam_names = cam_names

        self.action_key = "global_actions" if global_action else "actions"

        self.slices = []

        self.data = {}

        for cam_name in self.cam_names:
            self.data[cam_name] = []
        self.data["lang"] = []
        self.data["action"] = []

        i = 0
        self.envs_data = []
        for env in self.env_name:
            if 'PnP' in env:
                data_dir = os.path.join(data_directory, "kitchen_pnp", env)
            elif 'Door' in env:
                data_dir = os.path.join(data_directory, "kitchen_doors", env)
            elif 'Drawer' in env:
                data_dir = os.path.join(data_directory, "kitchen_drawer", env)
            elif 'Coffee' in env:
                data_dir = os.path.join(data_directory, "kitchen_coffee", env)
            elif 'Stove' in env:
                data_dir = os.path.join(data_directory, "kitchen_stove", env)
            else:
                raise ValueError(f"Unknown environment: {env}")

            data_dir = os.path.join(data_dir, os.listdir(data_dir)[0], "demo_gentex_im128_randcams.hdf5")

            env_data = h5py.File(data_dir, "r")
            env_data = env_data["data"]

            for demo in env_data:

                demo_length = env_data[demo].attrs["num_samples"]

                if demo_length - self.window_size < 0:
                    print(
                        f"Ignored short sequence #{i}: len={demo_length}, window={self.window_size}"
                    )
                else:
                    self.slices += [
                        (i, start, start + self.window_size)
                        for start in range(demo_length - self.window_size + 1)
                    ]  # slice indices follow convention [start, end)

                self.data["lang"].append(json.loads(env_data[demo].attrs["ep_meta"])["lang"])
                self.data["action"].append(env_data[demo][self.action_key][:, :self.action_dim])

                for cam_name in self.cam_names:
                    self.data[cam_name].append(env_data[demo]["obs"][f"{cam_name}_image"])

                i += 1

        cprint(f"Using dataset: {data_directory}", "green")
        cprint(f"Using action key: {self.action_key}", "blue")

    def get_seq_length(self, idx):
        pass
        # return self.demos[f"demo_{idx}"].attrs["num_samples"]

    def get_all_actions(self):
        result = []

        for action in self.data["action"]:
            result.append(torch.from_numpy(action))

        return torch.cat(result, dim=0).to(self.device)

    def get_all_observations(self):
        result = []

        for demo in self.demos:
            gripper_state = torch.from_numpy(
                self.demos[demo]["obs"]["robot0_gripper_qpos"][:, :1]
            )
            joint_pos = torch.from_numpy(self.demos[demo]["obs"]["robot0_joint_pos"][:])

            robot_state = torch.cat([gripper_state, joint_pos], dim=1)
            result.append(robot_state)

        return torch.cat(result, dim=0).to(self.device)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        action = torch.from_numpy(self.data["action"][i][start:end]).float()

        obs = {}

        for cam_name in self.cam_names:
            rgb = (
                torch.from_numpy(self.data[cam_name][i][start:start+1])
                .float()
                .permute(0, 3, 1, 2)
                / 255.0
            )
            obs[f"{cam_name}_image"] = rgb

        obs["lang"] = self.data["lang"][i]

        return obs, action, torch.ones(action.shape[0])  # TODO is this mask correct?
