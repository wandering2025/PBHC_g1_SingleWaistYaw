import multiprocessing
import time
import queue
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
import os
import traceback
from typing import List, Dict, Any

# 从重构后的训练脚本中导入核心逻辑函数
from humanoidverse.train_agent import launch_training

class ParallelTrainer:
    def __init__(self, tasks: List[Dict[str, Any]], config_path: str = "humanoidverse/config"):
        """
        并行训练器初始化
        
        :param tasks: 任务列表，每个字典代表一个独立的实验配置
        :param config_path: Hydra配置文件的相对路径
        """
        self.tasks = tasks
        self.config_path = config_path
        self.processes = []
        self.ipc_queues = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
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


if __name__ == "__main__":
    # 任务参数列表，每个字典代表一个独立的实验
    tasks = [
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
            "experiment_name": "debug_parallel_1",
            "seed": 1,
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