import logging
import os
import cv2
import random
import numpy as np
import torch
import wandb
import hydra
import multiprocessing as mp
from .base_sim import BaseSim
# from libero.libero.envs import *
from tqdm import tqdm
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.


class MultiTaskSim(BaseSim):
    def __init__(self,
                 num_episode,
                 max_step_per_episode,
                 task_suite: str,
                 use_eye_in_hand: bool,
                 seed,
                 device,
                 render,
                 n_cores,
                 use_multiprocessing=True):
        super().__init__(seed, device, render, n_cores)

        # according to the task_id, load the corresponding bddl file
        self.task_suite = task_suite

        self.use_eye_in_hand = use_eye_in_hand
        self.render = render

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0
        self.use_multiprocessing = use_multiprocessing

    def reverse_rgb_channels(self, test_img):

        test_img = test_img[::-1, ::-1, :]
        # cv2.imshow("test_img", test_img)
        # cv2.waitKey(0)

        return np.ascontiguousarray(test_img)

    def eval_agent(self,
                   contexts,
                   context_ind,
                   success,
                   episode_lengths,
                   pid,
                   cpu_set,
                   counter,
                   agent=None,
                   agent_config=None,
                   model_states=None):
        # Only set CPU affinity if using multiprocessing
        # if self.use_multiprocessing:
        #     print(os.getpid(), cpu_set)
        #     assign_process_to_cpu(os.getpid(), cpu_set)

        # Handle agent initialization based on input type
        if agent_config is not None:
            # Case 1: Initialize agent from config and states
            assert model_states is not None, "model_states must be provided when using agent_config"
            agent = hydra.utils.instantiate(agent_config)
            agent.recover_model_state(
                model_states['model'],
                model_states['scaler']
            )
        else:
            # Case 2: Use provided agent directly
            assert agent is not None, "Either agent or (agent_config + states) must be provided"

        # print(contexts)

        for i, context in enumerate(contexts):

            task_suite = benchmark.get_benchmark_dict()[self.task_suite]()

            task_bddl_file = task_suite.get_task_bddl_file_path(context)

            file_name = os.path.basename(task_bddl_file).split('.')[0]

            task_emb = self.task_embs[file_name].to(self.device).unsqueeze(0)

            # goal_images = self.goal_dicts[file_name]
            # goal_image = random.choice(goal_images)

            init_states = task_suite.get_task_init_states(context)

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }

            env = OffScreenRenderEnv(**env_args)

            agent.reset()
            env.seed(self.seed)
            env.reset()
            obs = env.set_init_state(init_state=init_states[context_ind[i]])

            # dummy actions all zeros for initial physics simulation
            dummy = np.zeros(7)
            dummy[-1] = -1.0  # set the last action to -1 to open the gripper
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                agentview_rgb = torch.from_numpy(obs["agentview_image"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.
                eye_in_hand_rgb = torch.from_numpy(obs["robot0_eye_in_hand_image"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.

                joint_state = obs["robot0_joint_pos"]
                gripper_state = obs["robot0_gripper_qpos"]

                robot_states = torch.from_numpy(np.concatenate([joint_state, gripper_state], axis=-1)).to(self.device).float().unsqueeze(0).unsqueeze(0)

                # save_path = os.path.join("/home/i53/student/wang/OCIL/OCIL", f"{self.task_suite}", "images")
                # img = env.sim.render(camera_name="frontview", width=1280, height=800)[..., ::-1]
                # img = np.flip(img, axis=0)
                # cv2.imwrite(os.path.join(save_path, f"agentview_{context}_{context_ind[i]}_{j}.png"), img)

                # agentview_rgb = self.reverse_rgb_channels(agentview_rgb)
                # eye_in_hand_rgb = self.reverse_rgb_channels(eye_in_hand_rgb)

                obs_dict = {"agentview_image": agentview_rgb,
                            "eye_in_hand_image": eye_in_hand_rgb,
                            "lang_emb": task_emb,
                            "robot_states": robot_states}

                action = agent.predict(obs_dict).cpu().numpy()
                obs, r, done, _ = env.step(action)

                # if self.render:
                # env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    episode_lengths[context, context_ind[i]] = j + 1
                    break
                    
            if success[context, context_ind[i]] == 0:
                episode_lengths[context, context_ind[i]] = self.max_step_per_episode

            if hasattr(counter, 'get_lock'):  # If it's a multiprocessing Value
                with counter.get_lock():
                    counter.value += 1
                    current_count = counter.value
            else:  # If it's a simple object with value attribute (single process)
                counter.value += 1
                current_count = counter.value
                counter.update()

            mask = episode_lengths.flatten() != 0
            completed_success = success.flatten()[mask]
            completed_lengths = episode_lengths.flatten()[mask]
            average_success = torch.mean(completed_success).item()
            average_episode_length = torch.mean(completed_lengths).item()
            log.info(f'completed_success {completed_success}')
            log.info(f'completed_lengths {completed_lengths}')
            log.info(f'average success rate: {average_success}')
            log.info(f'average episode length: {average_episode_length}')

            env.close()

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs

    def test_agent(self, agent, agent_config, cpu_set=None, epoch=None):
        logging.info("Start testing agent")

        if cpu_set is None:
            num_cpu = self.n_cores
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)
        
        if self.use_multiprocessing:
            log.info("there is {} cpus".format(num_cpu))
        else:
            log.info("not using multiprocessing, run on 1 cpu")

        if self.task_suite == "libero_90":
            num_tasks = 90
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        episode_lengths = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode

        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        if not self.use_multiprocessing:
            # Single process execution
            pbar = tqdm(total=all_runs, desc="Testing agent")
            counter = type('Counter', (), {'value': 0})()  # Simple counter object

            def update_pbar():
                pbar.update(1)
                
            counter.update = update_pbar  # Add update method to counter

            self.eval_agent(
                contexts=contexts,
                context_ind=context_ind,
                success=success,
                episode_lengths=episode_lengths,
                pid=0,
                cpu_set=set(cpu_set),
                counter=counter,
                agent=agent
            )
            pbar.close()
        else:
            repeat_num = all_runs // num_cpu
            repeat_res = all_runs % num_cpu

            workload_array = np.ones([num_cpu], dtype=int)
            workload_array[:repeat_res] += repeat_num
            workload_array[repeat_res:] = repeat_num

            assert np.sum(workload_array) == all_runs

            ind_workload = np.cumsum(workload_array)
            ind_workload = np.concatenate([[0], ind_workload])
            ###################################################################
            ctx = mp.get_context('spawn')
            processes_list = []

            all_runs = num_tasks * self.num_episode
            counter = ctx.Value('i', 0) #create a shared counter for progress bar
            pbar = tqdm(total=all_runs, desc="Testing agent")
            
            # Create shared memory state dictionaries for all models
            model_states = agent.get_model_state
            shared_states = {
                'model': {},
                'scaler': model_states[1]  # Assuming scaler is the 4th element
            }
    
            # Share memory for each state dictionary
            for key, tensor in model_states[0].items():
                shared_states['model'][key] = tensor.share_memory_()

            for i in range(self.n_cores):
                p = ctx.Process(target=self.eval_agent,
                                kwargs={  # Now passing single parameter
                                    "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                                    "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                                    "success": success,
                                    "episode_lengths": episode_lengths,
                                    "pid": i,
                                    "cpu_set": set(cpu_set[i:i + 1]),
                                    "counter": counter,
                                    "agent": None,
                                    "agent_config": agent_config,
                                    "model_states": shared_states,
                                },
                                )
                p.start()
                processes_list.append(p)
            
            # Monitor progress and update bar
            last_counter = 0
            while any(p.is_alive() for p in processes_list):
                if counter.value > last_counter:
                    pbar.update(counter.value - last_counter)
                    last_counter = counter.value

            [p.join() for p in processes_list]
            pbar.close()

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()

        print(f'success array {success.detach()}')

        custom_step = f"{epoch}_custom_step"
        wandb.define_metric(custom_step)
        wandb.define_metric(f"{epoch}_tasks_success", step_metric=custom_step)

        for num in range(num_tasks):
            log.info(f"Task {num}: {success_rate[num].item()}")

            wandb.log({custom_step: num,
                       f"{epoch}_tasks_success": success_rate[num].item()
                       })

        wandb.log({f"epoch{epoch}_average_success": average_success})
        log.info(f"Average success rate: {average_success}")
