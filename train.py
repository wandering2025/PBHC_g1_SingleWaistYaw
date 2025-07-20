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

# 从重构后的训练脚本中导入核心逻辑函数
from humanoidverse.train_agent import launch_training

def flatten_config(config, parent_key='', separator='.'):
    """递归展开嵌套配置字典为扁平结构"""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)

def run_process(config, ipc_queue):
    try:
        GlobalHydra.instance().clear()
        config_path = "humanoidverse/config"
        
        with initialize(config_path=config_path, version_base="1.1"):
            # 展平配置
            flat_config = flatten_config(config)
            overrides = [f"{key}={value}" for key, value in flat_config.items()]
            print(f"覆盖参数: {overrides}")
            
            # 组合配置
            subproc_cfg = compose(
                config_name="base",
                overrides=overrides,
                return_hydra_config=True
            )
            
            # 确保输出目录存在 - 现在直接从配置中获取
            if 'hydra' in subproc_cfg and 'run' in subproc_cfg.hydra and 'dir' in subproc_cfg.hydra.run:
                output_dir = subproc_cfg.hydra.run.dir
            elif 'experiment_name' in config:
                output_dir = f"outputs/{config['experiment_name']}"
            else:
                output_dir = "outputs/default"
            
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录设置为: {output_dir}")
            
            # 关键修复: 在设置 HydraConfig 之前更新输出目录
            # 1. 临时解除只读限制
            OmegaConf.set_readonly(subproc_cfg, False)
            
            # 2. 更新运行时输出目录
            OmegaConf.update(subproc_cfg, "hydra.runtime.output_dir", output_dir, merge=True)
            
            # 3. 确保 HydraConfig 被正确设置
            if not HydraConfig.initialized():
                HydraConfig.instance().set_config(subproc_cfg)
            
            # 4. 保持配置为可写状态，避免后续修改错误
            # 不再恢复只读状态
            
            # 调用训练函数
            launch_training(subproc_cfg, ipc_queue)
            
    except Exception as e:
        print(f"进程启动失败或在运行时发生错误: {e}")
        traceback.print_exc()
        ipc_queue.put({"error": str(e)})

def main():
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

    processes = []
    ipc_queues = []

    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"主脚本所在目录: {script_dir}")
    
    # 设置当前工作目录为脚本所在目录
    os.chdir(script_dir)
    print(f"当前工作目录: {os.getcwd()}")

    # 以编程方式初始化Hydra
    for i, task_overrides in enumerate(tasks):
        print(f"--- 准备任务 {i+1}: {task_overrides['experiment_name']} ---")
        
        # 为每个进程创建一个队列
        q = multiprocessing.Queue()
        ipc_queues.append(q)
        
        # 创建并启动进程
        p = multiprocessing.Process(target=run_process, args=(task_overrides, q))
        processes.append(p)
        p.start()

    # 监控循环
    try:
        active_processes = processes
        while active_processes:
            for i, p in enumerate(active_processes):
                # 从对应的队列中获取数据
                try:
                    data = ipc_queues[i].get_nowait()
                    if "error" in data:
                         print(f"❌ 收到来自任务 {tasks[i]['experiment_name']} 的错误: {data['error']}")
                    else:
                        print(f"✅ 收到 [任务 {i+1}] 数据: obs shape {data['obs_dict']['actor_obs'].shape}")
                except queue.Empty:
                    pass # 队列为空，正常

            # 移除已结束的进程
            active_processes = [p for p in active_processes if p.is_alive()]
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n用户中断，正在终止所有子进程...")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        print("所有进程已清理完毕。")


if __name__ == "__main__":
    # 使用 'spawn' 启动方法更安全、更跨平台，尤其是在使用CUDA时
    multiprocessing.set_start_method('spawn', force=True)
    main()