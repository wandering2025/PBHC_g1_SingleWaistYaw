import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. MODEL & PHYSICS DEFINITIONS (必须与您的训练脚本完全一致)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main_path(x) + self.shortcut(x))

class ParameterNetResNet(nn.Module):
    def __init__(self, input_dim: int, block_dims: list, num_dofs: int = 23, dropout_rate: float = 0.1):
        super(ParameterNetResNet, self).__init__()
        self.num_dofs = num_dofs
        first_block_dim = block_dims[0]
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, first_block_dim),
            nn.BatchNorm1d(first_block_dim),
            nn.ReLU()
        )
        res_layers = []
        current_dim = first_block_dim
        for h_dim in block_dims:
            res_layers.append(ResidualBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim
        self.residual_blocks = nn.Sequential(*res_layers)
        self.output_head = nn.Linear(current_dim, num_dofs * 3)

    def forward(self, x: torch.Tensor):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        raw_params = self.output_head(x)
        f_trans_raw, j_raw, tao_load_raw = torch.split(raw_params, self.num_dofs, dim=1)
        F_trans = f_trans_raw
        tao_load = tao_load_raw
        J = F.softplus(torch.tanh(j_raw))
        return F_trans, J, tao_load

def physics_compute_dof_alpha(f_c, r, F_trans, J, torque_a, tao_load, f_v, dof_vel):
    sign_dof_vel = torch.sign(dof_vel)
    sum_of_torques = -f_c * r * F_trans * sign_dof_vel - f_v * dof_vel + torque_a + tao_load
    dof_ang_acc_pre = sum_of_torques / (J + 1e-8)
    return dof_ang_acc_pre

def physics_compute_dof_vel(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    sign_dof_vel = torch.sign(dof_vel)
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    dof_vel_pre = numerator / (f_v + 1e-8)
    return dof_vel_pre

# --- 3. DATASET & COLLATOR (必须与您的训练脚本完全一致) ---
class PhysicsDataset(Dataset):
    def __init__(self, data: dict, input_keys: list, x_scaler: StandardScaler, window_size: int):
        self.input_keys = sorted(input_keys)
        self.all_keys = sorted(list(data.keys()))
        self.x_scaler = x_scaler
        self.window_size = window_size
        self.raw_data = data
        self._create_sample_indices()

    def _create_sample_indices(self):
        num_iterations = len(self.raw_data[self.all_keys[0]])
        seq_len = len(self.raw_data[self.all_keys[0]][0])
        self.sample_indices = []
        for i in range(num_iterations):
            for end_frame_t in range(self.window_size, seq_len):
                self.sample_indices.append((i, end_frame_t))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        iteration_idx, end_frame_t = self.sample_indices[idx]
        end_frame_t_minus_1 = end_frame_t - 1
        def get_sample(end_frame):
            start_frame = end_frame - self.window_size + 1
            x_window_raw = np.concatenate([
                self.raw_data[key][iteration_idx, start_frame:end_frame + 1] for key in self.input_keys
            ], axis=1)
            x_window_scaled = self.x_scaler.transform(x_window_raw)
            nn_input = torch.from_numpy(x_window_scaled.flatten().astype(np.float32))
            target_data = {key: torch.from_numpy(self.raw_data[key][iteration_idx, end_frame].astype(np.float32)) for key in self.all_keys}
            target_data['nn_input'] = nn_input
            return target_data
        sample_t_minus_1 = get_sample(end_frame_t_minus_1)
        sample_t = get_sample(end_frame_t)
        return {'t-1': sample_t_minus_1, 't': sample_t}

def collate_fn_physics(batch):
    collated_batch = {'t-1': {}, 't': {}}
    if not batch: return collated_batch
    for time_key in ['t-1', 't']:
        keys = batch[0][time_key].keys()
        for key in keys:
            tensors = [item[time_key][key] for item in batch]
            collated_batch[time_key][key] = torch.stack(tensors, dim=0)
    return collated_batch

# ==============================================================================
# 2. EVALUATION SCRIPT CONFIGURATION
# ==============================================================================
EXPERIMENT_DIR = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs/20250721_153042_dev_resnet_3TermLoss'
DATA_PATH = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/data_for_training/test_data.npy'

# 模型参数 (必须与您训练时使用的完全一致)
MODEL_PARAMS = {
    'input_dim': 1125,  # 75 features/frame * 10 frames
    'num_dofs': 23,
    'block_dims': [256], # 这是一个例子，请根据您训练的模型修改
    'dropout_rate': 0.25   # 这是一个例子，请根据您训练的模型修改
}
# 输入特征 (必须与您训练时使用的完全一致)
INPUT_KEYS = ['base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'dof_angular_acceleration']
WINDOW_SIZE = 15
DT = 0.005 # 1/200 Hz

NUM_EXAMPLES_TO_PRINT = 5

# ==============================================================================
# 3. MAIN EVALUATION LOGIC
# ==============================================================================
def main():
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = EXPERIMENT_DIR
        
    print(f"--- 开始评估实验 ---")
    print(f"日志目录: {log_dir}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载模型 ---
    model_path = os.path.join(log_dir, 'best_params_predictor.pth')
    model = ParameterNetResNet(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("模型加载成功。")

    # --- 加载数据标准化器 ---
    scaler_path = os.path.join(log_dir, 'x_scaler.gz')
    x_scaler = joblib.load(scaler_path)
    print("标准化器加载成功。")

    # --- 加载并准备数据 ---
    test_data = np.load(DATA_PATH, allow_pickle=True).item()
    test_dataset = PhysicsDataset(test_data, INPUT_KEYS, x_scaler, WINDOW_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, collate_fn=collate_fn_physics, num_workers=4)
    print(f"测试数据加载成功，共 {len(test_dataset)} 个样本。\n")
    
    # --- in the main() function of eval.py ---

    r_tensor = torch.tensor([
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 
        0.025, 
        0.025, 0.025, 0.025, 0.025, 0.018, 
        0.025, 0.025, 0.025, 0.025, 0.018
    ], dtype=torch.float32).to(device)

    np.set_printoptions(precision=4, suppress=True)
    criterion = nn.SmoothL1Loss(reduction='none') # 使用'none'以便我们能分别计算和聚合
    mae_criterion = nn.L1Loss()

    # --- 初始化用于聚合指标的变量 ---
    total_samples = 0
    total_loss_accel, total_loss_vel, total_loss_consist = 0.0, 0.0, 0.0
    total_vel_mae = 0.0

    print(f"--- 详细打印 {NUM_EXAMPLES_TO_PRINT} 个具体样本 ---")
# --- in the main() function of eval_time_window.py ---

    # --- 1. 详细打印一个具体的样本 ---
    # 您可以通过修改这个索引来选择打印测试集中的第几个样本
    SAMPLE_IDX_TO_PRINT = 0 
    print(f"--- 详细打印样本索引 {SAMPLE_IDX_TO_PRINT} ---")

    with torch.no_grad():
        # 直接从dataset获取一个样本，避免加载整个批次
        single_sample = test_dataset[SAMPLE_IDX_TO_PRINT]
        
        # 手动为该样本的t时刻数据添加batch维度并移动到device
        batch_t = {k: v.unsqueeze(0).to(device) for k, v in single_sample['t'].items()}
        
        # 使用t时刻的输入进行一次前向传播
        F_trans_pred, J_pred, tao_load_pred = model(batch_t['nn_input'])
        
        accel_pred = physics_compute_dof_alpha(
            f_c=batch_t['friction_coeffs'], r=r_tensor, F_trans=F_trans_pred, J=J_pred,
            torque_a=batch_t['torque'], tao_load=tao_load_pred,
            f_v=batch_t['viscous_friction_coeffs'], dof_vel=batch_t['dof_vel']
        )
        
        # --- 开始打印详细信息 ---
        s =4# 为保持可读性，只打印向量的前4个元素
        print(f"\n--- [ 样本 {SAMPLE_IDX_TO_PRINT} 详细分析 ] ---")
        print(f"  输入状态 (真值):")
        print(f"    - Base Angular Vel : {batch_t['base_angular_vel'][0, :3].cpu().numpy()}")
        print(f"    - Projected Gravity: {batch_t['projected_gravity'][0, :3].cpu().numpy()}")
        print(f"    - Dof Pos (rad)    : {batch_t['dof_pos'][0, :s].cpu().numpy()}")
        print(f"    - Dof Vel (rad/s)  : {batch_t['dof_vel'][0, :s].cpu().numpy()}")
        print(f"    - Dof Acc (rad/s^2): {batch_t['dof_angular_acceleration'][0, :s].cpu().numpy()}")
        
        print(f"  预测的物理参数:")
        print(f"    - 惯性 J             : {J_pred[0, :s].cpu().numpy()}")
        print(f"    - 力 F_trans         : {F_trans_pred[0, :s].cpu().numpy()}")
        print(f"    - 负载 tao_load      : {tao_load_pred[0, :s].cpu().numpy()}")
            
        print(f"  角加速度对比:")
        print(f"    - 预测值 (rad/s^2)   : {accel_pred[0, :s].cpu().numpy()}")
        print(f"    - 真实值 (rad/s^2)   : {batch_t['dof_angular_acceleration'][0, :s].cpu().numpy()}")

    print(f"\n--- 在整个测试集上计算总体指标 ---")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch_tm1 = {k: v.to(device) for k, v in batch['t-1'].items()}
            batch_t = {k: v.to(device) for k, v in batch['t'].items()}
            
            # --- 在 t-1 和 t 时刻进行两次前向传播 ---
            F_trans_tm1, J_tm1, tao_load_tm1 = model(batch_tm1['nn_input'])
            accel_pred_tm1 = physics_compute_dof_alpha(f_c=batch_tm1['friction_coeffs'], r=r_tensor, F_trans=F_trans_tm1, J=J_tm1, torque_a=batch_tm1['torque'], tao_load=tao_load_tm1, f_v=batch_tm1['viscous_friction_coeffs'], dof_vel=batch_tm1['dof_vel'])
            vel_pred_tm1 = physics_compute_dof_vel(f_c=batch_tm1['friction_coeffs'], r=r_tensor, F_trans=F_trans_tm1, J=J_tm1, dof_ang_acc=batch_tm1['dof_angular_acceleration'], torque_a=batch_tm1['torque'], tao_load=tao_load_tm1, f_v=batch_tm1['viscous_friction_coeffs'], dof_vel=batch_tm1['dof_vel'])
            
            F_trans_t, J_t, tao_load_t = model(batch_t['nn_input'])
            accel_pred_t = physics_compute_dof_alpha(f_c=batch_t['friction_coeffs'], r=r_tensor, F_trans=F_trans_t, J=J_t, torque_a=batch_t['torque'], tao_load=tao_load_t, f_v=batch_t['viscous_friction_coeffs'], dof_vel=batch_t['dof_vel'])
            vel_pred_t = physics_compute_dof_vel(f_c=batch_t['friction_coeffs'], r=r_tensor, F_trans=F_trans_t, J=J_t, dof_ang_acc=batch_t['dof_angular_acceleration'], torque_a=batch_t['torque'], tao_load=tao_load_t, f_v=batch_t['viscous_friction_coeffs'], dof_vel=batch_t['dof_vel'])

            # --- 计算所有指标 ---
            # 1. 三个损失分量 (unweighted)
            loss_accel = criterion(accel_pred_t, batch_t['dof_angular_acceleration']).mean()
            loss_vel = criterion(vel_pred_t, batch_t['dof_vel']).mean()
            loss_consistency = criterion(vel_pred_t - vel_pred_tm1, accel_pred_tm1 * DT).mean()
            
            # 2. 最终的速度预测平均绝对误差 (MAE)
            vel_mae = mae_criterion(vel_pred_t, batch_t['dof_vel'])
            
            # --- 累加指标 ---
            batch_size = batch_t['nn_input'].size(0)
            total_samples += batch_size
            total_loss_accel += loss_accel.item() * batch_size
            total_loss_vel += loss_vel.item() * batch_size
            total_loss_consist += loss_consistency.item() * batch_size
            total_vel_mae += vel_mae.item() * batch_size
            
            # 打印进度
            sys.stdout.write(f"\r处理进度: {i+1}/{len(test_loader)}")
            sys.stdout.flush()

    # --- 计算并打印最终平均指标 ---
    print("\n\n--- 评估结果汇总 ---")
    avg_loss_accel = total_loss_accel / total_samples
    avg_loss_vel = total_loss_vel / total_samples
    avg_loss_consist = total_loss_consist / total_samples
    avg_vel_mae = total_vel_mae / total_samples

    print(f"最终物理指标:")
    print(f"  - 角速度预测平均绝对误差 (MAE): {avg_vel_mae:.4f} rad/s")
    print("\n损失函数分量 (在测试集上的表现):")
    print(f"  - 角加速度损失 (loss_accel): {avg_loss_accel:.4f}")
    print(f"  - 角速度损失 (loss_vel): {avg_loss_vel:.4f}")
    print(f"  - 内部一致性损失 (loss_consist): {avg_loss_consist:.4f}")
    print("\n--- 评估结束 ---")

if __name__ == '__main__':
    # Fill in r_tensor values before calling main
    r_tensor_values = [
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 0.025,
        0.025, 0.025, 0.025, 0.025, 0.018, 0.025, 0.025, 0.025, 0.025, 0.018
    ]
    # This is a bit of a workaround to pass r_tensor to main, clean way is to load it inside main.
    # For now, we redefine it inside the main function where it's used.
    def main_wrapper():
        global r_tensor
        r_tensor = torch.tensor(r_tensor_values, dtype=torch.float32)
        # The main function expects r_tensor to be defined in its scope, let's redefine it there
        # for clarity and to avoid global variables.
        main()
        
    main()