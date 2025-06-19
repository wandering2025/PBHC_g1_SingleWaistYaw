import os
import sys
from pathlib import Path

paths_to_remove = [
    '/home/bbw/ASAPx1',
    '/home/bbw/ASAPx1/isaac_utils',
    # 移除任何不必要的或重复的 PBHC 相关路径，避免混淆
    '/home/bbw/PBHC/humanoidverse/isaac_utils/isaac_utils/__init__.py', # 文件路径
    '/home/bbw/PBHC/humanoidverse/isaac_utils', # 这是 isaac_utils 的直接父目录，但如果 PBHC_ROOT 包含了它，就不需要单独添加
    # '/home/bbw', # 如果之前有这个，也要移除，因为它太宽泛
    '/home/bbw/PBHC/humanoidverse' # 移除脚本所在的目录，避免它作为顶层路径被优先查找
]

# 倒序遍历 sys.path 以安全地移除元素
for p in reversed(sys.path):
    if p in paths_to_remove:
        sys.path.remove(p)

pbhc_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
isaac_utils_actual_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'isaac_utils'))
#isaac_utils_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'isaac_utils'))
#pbhc_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if pbhc_project_root not in sys.path:
    sys.path.insert(0, pbhc_project_root)
if isaac_utils_actual_parent_path not in sys.path:
    # 插入到 pbhc_project_root 之后
    sys.path.insert(1 if len(sys.path) > 0 and sys.path[0] == pbhc_project_root else 0, isaac_utils_actual_parent_path)

print(f"DEBUG: PBHC project root added to sys.path: {pbhc_project_root}")
print(f"DEBUG: sys.path after modification and cleanup: {sys.path}")

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
from loguru import logger

from humanoidverse.utils.devtool import *


from humanoidverse.utils.config_utils import *  # noqa: E402, F403
@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    # import ipdb; ipdb.set_trace()
    simulator_type = config.simulator['_target_'].split('.')[-1]
    # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  
        
        # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacGym':
        import isaacgym  # noqa: F401


    # have to import torch after isaacgym
    import torch  # noqa: E402
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge
        
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    if config.use_wandb:
        import wandb
        project_name = f"{config.project_name}"
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(project=project_name, 
                entity=config.wandb.wandb_entity,
                name=run_name,
                sync_tensorboard=True,
                config=unresolved_conf,
                dir=wandb_dir)
    
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pre_process_config(config)

    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env: BaseEnv = instantiate(config=config.env, device=device)


    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving config file to {experiment_save_dir}")
    with open(experiment_save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)

    algo: BaseAlgo = instantiate(device=device, env=env, config=config.algo, log_dir=experiment_save_dir)
    algo.setup()
    if config.checkpoint is not None:
        algo.load(config.checkpoint)

    # handle saving config
    algo.learn()

    if simulator_type == 'IsaacSim':
        simulation_app.close()

if __name__ == "__main__":
    main()
