import multiprocessing
import time
import queue
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
import isaacgym
from humanoidverse.agents.modules.ppo_modules import *

import os
import traceback
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from config import get_config

# 从重构后的训练脚本中导入核心逻辑函数
from humanoidverse.train_agent import launch_training

class ParallelTrainer:
    def __init__(self, tasks: List[Dict[str, Any]], config_path: str = "humanoidverse/config"):
        """
        并行训练器初始化
        
        :param tasks: 任务列表，每个字典代表一个独立的实验配置
        :param config_path: Hydra配置文件的相对路径
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tasks = tasks
        self.config_path = config_path
        self.processes = []
        self.ipc_queues = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # self.cfg = OmegaConf.create(get_config())
        # 创建模型和优化器
        # self._setup_models_and_optimizer()
        # 设置当前工作目录
        os.chdir(self.script_dir)
        print(f"当前工作目录: {os.getcwd()}")
        
        # 设置多进程启动方法
        multiprocessing.set_start_method('spawn', force=True)
    
    @staticmethod
    def flatten_config(config: Dict[str, Any], parent_key: str = '', separator: str = '.') -> Dict[str, Any]:
        """
        递归展开嵌套配置字典为扁平结构
        
        :param config: 嵌套配置字典
        :param parent_key: 父键名
        :param separator: 键分隔符
        :return: 扁平化的配置字典
        """
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ParallelTrainer.flatten_config(v, new_key, separator=separator).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def run_process(config: Dict[str, Any], ipc_queue: multiprocessing.Queue, config_path: str):
        """
        子进程执行函数
        
        :param config: 任务配置
        :param ipc_queue: 进程间通信队列
        :param config_path: Hydra配置文件的相对路径
        """
        try:
            GlobalHydra.instance().clear()
            
            with initialize(config_path=config_path, version_base="1.1"):
                # 展平配置
                flat_config = ParallelTrainer.flatten_config(config)
                overrides = [f"{key}={value}" for key, value in flat_config.items()]
                print(f"覆盖参数: {overrides}")
                
                # 组合配置
                subproc_cfg = compose(
                    config_name="base",
                    overrides=overrides,
                    return_hydra_config=True
                )
                
                # 确保输出目录存在
                if 'hydra' in subproc_cfg and 'run' in subproc_cfg.hydra and 'dir' in subproc_cfg.hydra.run:
                    output_dir = subproc_cfg.hydra.run.dir
                elif 'experiment_name' in config:
                    output_dir = f"outputs/{config['experiment_name']}"
                else:
                    output_dir = "outputs/default"
                
                os.makedirs(output_dir, exist_ok=True)
                print(f"输出目录设置为: {output_dir}")
                
                # 更新运行时配置
                OmegaConf.set_readonly(subproc_cfg, False)
                OmegaConf.update(subproc_cfg, "hydra.runtime.output_dir", output_dir, merge=True)
                
                # 确保 HydraConfig 被正确设置
                if not HydraConfig.initialized():
                    HydraConfig.instance().set_config(subproc_cfg)
                
                # 调用训练函数
                launch_training(subproc_cfg, ipc_queue)
                
        except Exception as e:
            print(f"进程启动失败或在运行时发生错误: {e}")
            traceback.print_exc()
            ipc_queue.put({"error": str(e)})
    
    def start_training(self):
        """启动所有训练任务"""
        print(f"主脚本所在目录: {self.script_dir}")
        
        for i, task_overrides in enumerate(self.tasks):
            print(f"--- 准备任务 {i+1}: {task_overrides['experiment_name']} ---")
            
            # 为每个进程创建一个队列
            q = multiprocessing.Queue()
            self.ipc_queues.append(q)
            
            # 创建并启动进程
            p = multiprocessing.Process(
                target=self.run_process, 
                args=(task_overrides, q, self.config_path)
            )
            self.processes.append(p)
            p.start()
    
    def monitor(self):
        """监控训练进程状态"""
        try:
            active_processes = self.processes.copy()
            while active_processes:
                for i, p in enumerate(active_processes):
                    # 从对应的队列中获取数据
                    try:
                        data = self.ipc_queues[i].get_nowait()
                        if "error" in data:
                            print(f"❌ 收到来自任务 {self.tasks[i]['experiment_name']} 的错误: {data['error']}")
                        else:
                            print(f"✅ 收到 [任务 {i+1}] 数据: obs shape {data['obs_dict']['actor_obs'].shape}")
                            # obs_dict = data['obs_dict']
                            # self.storage = data['storage']
                            # loss_dict = self._training_step()
                    except queue.Empty:
                        pass  # 队列为空，正常
                
                # 移除已结束的进程
                active_processes = [p for p in active_processes if p.is_alive()]
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n用户中断，正在终止所有子进程...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理所有进程资源"""
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        print("所有进程已清理完毕。")
    
    def run(self):
        """执行完整训练流程"""
        self.start_training()
        self.monitor()

    def _setup_models_and_optimizer(self):
        print("-" * 150)
        print(self.cfg['module_dict']['actor'])
        actor_kwargs = {
            "obs_dim_dict": self.cfg['algo_obs_dim_dict'],
            "module_config_dict": self.cfg['module_dict']['actor'],
            "num_actions": self.cfg['num_actions'],
            "init_noise_std": self.cfg['init_noise_std']
        }
        critic_kwargs = {
            "obs_dim_dict": self.cfg['algo_obs_dim_dict'],
            "module_config_dict": self.cfg['module_dict']['critic'],
        }
        self.actor = PPOActor(
            **actor_kwargs
        ).to(self.device)

        self.critic = PPOCritic(
            **critic_kwargs
        ).to(self.device)

            
        print(self.actor)
        print(self.critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_learning_rate)

    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for policy_state_dict in generator:
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict
    
    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict['Value'] = 0
        loss_dict['Surrogate'] = 0
        loss_dict['Entropy'] = 0
        loss_dict['L2C2_Value'] = 0
        loss_dict['L2C2_Policy'] = 0
        return loss_dict
    
    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_ppo(policy_state_dict, loss_dict)
        return loss_dict

    def _actor_act_step(self, obs_dict):
        return self.actor.act(obs_dict["actor_obs"])
    
    def _critic_eval_step(self, obs_dict):
        return self.critic.evaluate(obs_dict["critic_obs"])
    
    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).sum(dim=-1).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).sum(dim=-1).mean()

        entropy_loss = entropy_batch.mean()
        
        # L2C2 smooth
        if self.cfg_l2c2 is not None and self.cfg_l2c2.enable:
            lam_value = self.cfg_l2c2.lambda_value
            lam_policy = self.cfg_l2c2.lambda_policy
            actor_obs, next_actor_obs = policy_state_dict['actor_obs'], policy_state_dict['next_actor_obs']
            critic_obs, next_critic_obs = policy_state_dict['critic_obs'], policy_state_dict['next_critic_obs']
            
            u = torch.rand(*actor_obs.shape[:-1],1, device=self.device)*2-1
            u_actor_obs = actor_obs + u*(next_actor_obs-actor_obs)
            u_critic_obs = critic_obs + u*(next_critic_obs-critic_obs)
            
            u_mu = self.actor.act_inference(u_actor_obs)
            u_value = self.critic.evaluate(u_critic_obs)
            
            
            l2c2_value_loss = lam_value * (value_batch - u_value).pow(2).mean()
            l2c2_policy_loss = lam_policy * (actions_batch - u_mu).pow(2).mean()
            # breakpoint()
        else:
            l2c2_value_loss = torch.tensor(0.0, device=self.device)
            l2c2_policy_loss = torch.tensor(0.0, device=self.device)
        
        
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss + l2c2_policy_loss
        
        critic_loss = self.value_loss_coef * value_loss + l2c2_value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # print("skip backward")
        actor_loss.backward()
        critic_loss.backward()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['L2C2_Value'] += l2c2_value_loss.item()
        loss_dict['L2C2_Policy'] += l2c2_policy_loss.item()
        return loss_dict
    
if __name__ == "__main__":
    # 任务参数列表，每个字典代表一个独立的实验
    tasks = [
        {
            "+simulator": "isaacgym",
            "+exp": "motion_tracking",
            "+terrain": "terrain_locomotion_plane",
            "project_name": "MotionTracking",
            "num_envs": 1,
            "+obs": "motion_tracking/main",
            "+robot": "g1/g1_23dof_lock_wrist",
            "+domain_rand": "main",
            "+rewards": "motion_tracking/main",
            "experiment_name": "debug_parallel_1",
            "seed": 128,
            "+device": "cuda:0",
            "robot.motion.motion_file": "example/motion_data/Horse-stance_pose.pkl",
            "hydra.run.dir": "outputs/debug_parallel_1"
        },
        {
            "+simulator": "isaacgym",
            "+exp": "motion_tracking",
            "+terrain": "terrain_locomotion_plane",
            "project_name": "MotionTracking",
            "num_envs": 128,
            "+obs": "motion_tracking/main",
            "+robot": "g1/g1_23dof_lock_wrist",
            "+domain_rand": "main",
            "+rewards": "motion_tracking/main",
            "experiment_name": "debug_parallel_2",
            "seed": 2,
            "+device": "cuda:0",
            "robot.motion.motion_file": "example/motion_data/Horse-stance_pose.pkl",
            "hydra.run.dir": "outputs/debug_parallel_2"
        },
        # 在这里可以添加更多任务...
    ]
    
    # 创建并行训练器并运行
    trainer = ParallelTrainer(tasks)
    trainer.run()