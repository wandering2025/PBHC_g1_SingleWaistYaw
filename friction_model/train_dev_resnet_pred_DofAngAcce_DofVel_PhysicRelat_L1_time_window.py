import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import datetime
import sys
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1a. Residual Block Definition (Unchanged) ---
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

# --- 1b. Parameter Prediction Network (Unchanged, uses flattened input) ---
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

# --- 2. Differentiable Physics Models (Both are needed) ---
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

# --- 3. Custom Dataset (Provides t and t-1 data) ---
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
        print(f"Generating paired (t-1, t) sample indices... (Window Size: {self.window_size})")
        for i in range(num_iterations):
            # The earliest `end_frame_t` can be is `window_size`
            for end_frame_t in range(self.window_size, seq_len):
                self.sample_indices.append((i, end_frame_t))
        print(f"Successfully created {len(self.sample_indices)} total samples.")

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

# --- 4. Custom Collate Function (MODIFIED for nested dictionary) ---
def collate_fn_physics(batch):
    collated_batch = {'t-1': {}, 't': {}}
    if not batch: return collated_batch
    for time_key in ['t-1', 't']:
        keys = batch[0][time_key].keys()
        for key in keys:
            tensors = [item[time_key][key] for item in batch]
            collated_batch[time_key][key] = torch.stack(tensors, dim=0)
    return collated_batch

# --- 5. Training Loop (MODIFIED for 3-Term Loss and Detailed Logging) ---
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience, dt, w_accel, w_vel, w_consist):
    best_model_path = os.path.join(session_save_dir, 'best_params_predictor.pth')
    min_val_loss = float('inf')
    epochs_no_improve = 0
    r_tensor = torch.tensor([
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 0.025,
        0.025, 0.025, 0.025, 0.025, 0.018, 0.025, 0.025, 0.025, 0.025, 0.018
    ], dtype=torch.float32).to(device)

    print(f"\n--- Starting Training with 3-Term Loss (w_accel={w_accel}, w_vel={w_vel}, w_consist={w_consist}) ---")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch_tm1 = {k: v.to(device) for k, v in batch['t-1'].items()}
            batch_t = {k: v.to(device) for k, v in batch['t'].items()}
            optimizer.zero_grad()
            
            # --- Forward pass for t-1 ---
            F_trans_tm1, J_tm1, tao_load_tm1 = model(batch_tm1['nn_input'])
            accel_pred_tm1 = physics_compute_dof_alpha(f_c=batch_tm1['friction_coeffs'], r=r_tensor, F_trans=F_trans_tm1, J=J_tm1, torque_a=batch_tm1['torque'], tao_load=tao_load_tm1, f_v=batch_tm1['viscous_friction_coeffs'], dof_vel=batch_tm1['dof_vel'])
            vel_pred_tm1 = physics_compute_dof_vel(f_c=batch_tm1['friction_coeffs'], r=r_tensor, F_trans=F_trans_tm1, J=J_tm1, dof_ang_acc=batch_tm1['dof_angular_acceleration'], torque_a=batch_tm1['torque'], tao_load=tao_load_tm1, f_v=batch_tm1['viscous_friction_coeffs'], dof_vel=batch_tm1['dof_vel'])
            
            # --- Forward pass for t ---
            F_trans_t, J_t, tao_load_t = model(batch_t['nn_input'])
            accel_pred_t = physics_compute_dof_alpha(f_c=batch_t['friction_coeffs'], r=r_tensor, F_trans=F_trans_t, J=J_t, torque_a=batch_t['torque'], tao_load=tao_load_t, f_v=batch_t['viscous_friction_coeffs'], dof_vel=batch_t['dof_vel'])
            vel_pred_t = physics_compute_dof_vel(f_c=batch_t['friction_coeffs'], r=r_tensor, F_trans=F_trans_t, J=J_t, dof_ang_acc=batch_t['dof_angular_acceleration'], torque_a=batch_t['torque'], tao_load=tao_load_t, f_v=batch_t['viscous_friction_coeffs'], dof_vel=batch_t['dof_vel'])

            # --- 3-TERM LOSS CALCULATION ---
            loss_accel = criterion(accel_pred_t, batch_t['dof_angular_acceleration'])
            loss_vel = criterion(vel_pred_t, batch_t['dof_vel'])
            loss_consistency = criterion(vel_pred_t - vel_pred_tm1, accel_pred_tm1 * dt)
            
            total_loss = (w_accel * loss_accel + 
                          w_vel * loss_vel + 
                          w_consist * loss_consistency)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += total_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation loop ---
        model.eval()
        total_val_loss = 0
        # For logging individual components
        val_loss_accel_agg, val_loss_vel_agg, val_loss_consist_agg = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_tm1 = {k: v.to(device) for k, v in batch['t-1'].items()}
                batch_t = {k: v.to(device) for k, v in batch['t'].items()}

                F_trans_tm1, J_tm1, tao_load_tm1 = model(batch_tm1['nn_input'])
                accel_pred_tm1 = physics_compute_dof_alpha(f_c=batch_tm1['friction_coeffs'], r=r_tensor, F_trans=F_trans_tm1, J=J_tm1, torque_a=batch_tm1['torque'], tao_load=tao_load_tm1, f_v=batch_tm1['viscous_friction_coeffs'], dof_vel=batch_tm1['dof_vel'])
                vel_pred_tm1 = physics_compute_dof_vel(f_c=batch_tm1['friction_coeffs'], r=r_tensor, F_trans=F_trans_tm1, J=J_tm1, dof_ang_acc=batch_tm1['dof_angular_acceleration'], torque_a=batch_tm1['torque'], tao_load=tao_load_tm1, f_v=batch_tm1['viscous_friction_coeffs'], dof_vel=batch_tm1['dof_vel'])
                
                F_trans_t, J_t, tao_load_t = model(batch_t['nn_input'])
                accel_pred_t = physics_compute_dof_alpha(f_c=batch_t['friction_coeffs'], r=r_tensor, F_trans=F_trans_t, J=J_t, torque_a=batch_t['torque'], tao_load=tao_load_t, f_v=batch_t['viscous_friction_coeffs'], dof_vel=batch_t['dof_vel'])
                vel_pred_t = physics_compute_dof_vel(f_c=batch_t['friction_coeffs'], r=r_tensor, F_trans=F_trans_t, J=J_t, dof_ang_acc=batch_t['dof_angular_acceleration'], torque_a=batch_t['torque'], tao_load=tao_load_t, f_v=batch_t['viscous_friction_coeffs'], dof_vel=batch_t['dof_vel'])

                loss_accel_val = criterion(accel_pred_t, batch_t['dof_angular_acceleration'])
                loss_vel_val = criterion(vel_pred_t, batch_t['dof_vel'])
                loss_consistency_val = criterion(vel_pred_t - vel_pred_tm1, accel_pred_tm1 * dt)
                
                total_loss_val = (w_accel * loss_accel_val + 
                                  w_vel * loss_vel_val + 
                                  w_consist * loss_consistency_val)
                
                total_val_loss += total_loss_val.item()
                # Aggregate unscaled losses for logging
                val_loss_accel_agg += loss_accel_val.item()
                val_loss_vel_agg += loss_vel_val.item()
                val_loss_consist_agg += loss_consistency_val.item()

        avg_val_loss = total_val_loss / len(val_loader)
        # Calculate average for individual components
        avg_loss_accel_val = val_loss_accel_agg / len(val_loader)
        avg_loss_vel_val = val_loss_vel_agg / len(val_loader)
        avg_loss_consist_val = val_loss_consist_agg / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")
        # --- MODIFIED: Detailed Logging ---
        print(f"  └─ Val Loss Components (Weighted): Accel: {avg_loss_accel_val * w_accel:.4f}, Vel: {avg_loss_vel_val * w_vel:.4f}, Consist: {avg_loss_consist_val * w_consist:.4f}")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best validation loss. Model saved.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break
        scheduler.step(avg_val_loss)

    print("\n--- Training Finished ---")
    print(f"Best model saved at: {best_model_path}")

# --- 6. Main Execution Block ---
def main():
    # --- Configuration ---
    input_keys = ['base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'dof_angular_acceleration']
    num_dofs = 23
    window_size = 10 
    input_dim_per_frame = sum([3, 3, 23, 23, 23]) # 75
    input_dim = input_dim_per_frame * window_size
    
    # --- Architecture and Hyperparameters ---
    block_dims = [512, 1024,1024, 512] 
    dropout_rate = 0.2
    
    # --- MODIFIED: Hyperparameters for 3-Term Loss ---
    dt = 1 / 200 
    # Start with weights that prioritize the most stable loss (accel)
    loss_weight_accel = 1.0
    loss_weight_vel = 0.1
    loss_weight_consistency = 0.1

    # Start with a conservative learning rate due to complex loss
    learning_rate = 1e-5
    weight_decay = 1e-4
    batch_size = 2048
    num_epochs = 5000
    patience = 150
    scheduler_patience = 40
    
    # --- Paths and Session Management ---
    log_base_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs'
    data_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/data_for_training'
    experiment_name = 'dev_resnet_3TermLoss'
    # ... (The rest of the path/session management code is unchanged) ...
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_save_dir = os.path.join(log_base_dir, f"{timestamp}_{experiment_name}")
    os.makedirs(session_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading and Standardization ---
    print("Loading pre-split data...")
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True).item()
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True).item()
    print("Fitting StandardScaler on training data...")
    sorted_input_keys = sorted(input_keys)
    full_train_x_raw = np.concatenate([train_data[key] for key in sorted_input_keys], axis=2)
    num_features = full_train_x_raw.shape[2]
    full_train_x_reshaped = full_train_x_raw.reshape(-1, num_features)
    x_scaler = StandardScaler()
    x_scaler.fit(full_train_x_reshaped)
    del full_train_x_raw, full_train_x_reshaped
    joblib.dump(x_scaler, os.path.join(session_save_dir, 'x_scaler.gz'))
    print(f"Input feature scaler (x_scaler) saved to {session_save_dir}")
    
    # --- Create Datasets and DataLoaders ---
    train_dataset = PhysicsDataset(train_data, input_keys, x_scaler, window_size)
    val_dataset = PhysicsDataset(val_data, input_keys, x_scaler, window_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_physics, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_physics, num_workers=4, pin_memory=True)

    # --- Initialize Model, Criterion, and Optimizer ---
    model = ParameterNetResNet(input_dim=input_dim, 
                               block_dims=block_dims, 
                               num_dofs=num_dofs, 
                               dropout_rate=dropout_rate).to(device)
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    
    print("\n--- Starting run for experiment:", experiment_name, "---")
    print(f"Input dim per frame: {input_dim_per_frame} | Window: {window_size} -> Total Input Dim: {input_dim}")
    print(f"Network Architecture (block_dims): {block_dims}")
    print("\nModel Structure:")
    print(model)
    
    # --- Start Training ---
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
          session_save_dir, num_epochs, patience, dt, 
          loss_weight_accel, loss_weight_vel, loss_weight_consistency)

if __name__ == '__main__':
    main()