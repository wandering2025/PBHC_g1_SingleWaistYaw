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
            nn.Linear(in_dim, out_dim), 
            nn.BatchNorm1d(out_dim), 
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main_path(x) + self.shortcut(x))

# --- 1b. Temporal Parameter Prediction Network with 1D-CNN (Unchanged) ---
class TemporalParameterNet(nn.Module):
    def __init__(self, input_dim_per_frame: int, cnn_out_channels: int, block_dims: list, num_dofs: int = 23, dropout_rate: float = 0.1):
        super(TemporalParameterNet, self).__init__()
        self.num_dofs = num_dofs
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim_per_frame, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        res_layers = []
        current_dim = cnn_out_channels
        for h_dim in block_dims:
            res_layers.append(ResidualBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim
        self.residual_blocks = nn.Sequential(*res_layers)
        self.output_head = nn.Linear(current_dim, num_dofs * 3)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.temporal_encoder(x)
        x = x.squeeze(-1)
        x = self.residual_blocks(x)
        raw_params = self.output_head(x)
        f_trans_raw, j_raw, tao_load_raw = torch.split(raw_params, self.num_dofs, dim=1)
        F_trans = f_trans_raw
        tao_load = tao_load_raw
        J = F.softplus(torch.tanh(j_raw))
        return F_trans, J, tao_load

# --- 2. Differentiable Physics Model (Unchanged) ---
def physics_compute_dof_alpha(f_c, r, F_trans, J, torque_a, tao_load, f_v, dof_vel):
    sign_dof_vel = torch.sign(dof_vel)
    sum_of_torques = -f_c * r * F_trans * sign_dof_vel - f_v * dof_vel + torque_a + tao_load
    dof_ang_acc_pre = sum_of_torques / (J + 1e-8)
    return dof_ang_acc_pre

# --- 3. Custom Dataset (MODIFIED for Temporal Consistency Loss) ---
class PhysicsDataset(Dataset):
    def __init__(self, data: dict, input_keys: list, x_scaler: StandardScaler, window_size: int):
        self.input_keys = sorted(input_keys)
        self.all_keys = sorted(list(data.keys()))
        self.x_scaler = x_scaler
        self.window_size = window_size
        self._create_windows(data)

    def _create_windows(self, data: dict):
        x_raw_all = np.concatenate([data[key] for key in self.input_keys], axis=2, dtype=np.float32)
        num_iterations, seq_len, _ = x_raw_all.shape
        self.windows = []
        self.targets_t = []
        self.targets_t_plus_1 = [] # NEW: To store data for the next timestep
        
        print(f"Generating sliding windows... (Window Size: {self.window_size})")
        for i in range(num_iterations):
            # Loop must stop at seq_len - 2 to allow access to t+1 data.
            for end_frame in range(self.window_size - 1, seq_len - 1):
                start_frame = end_frame - self.window_size + 1
                self.windows.append(x_raw_all[i, start_frame:end_frame + 1, :])
                # Target 't' is the data at the end of the window
                self.targets_t.append({key: data[key][i, end_frame] for key in self.all_keys})
                # Target 't+1' is the data for the next frame
                self.targets_t_plus_1.append({key: data[key][i, end_frame + 1] for key in self.all_keys})
        print(f"Successfully created {len(self.windows)} total samples.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        x_window_raw = self.windows[idx]
        x_window_scaled = self.x_scaler.transform(x_window_raw)
        nn_input = torch.from_numpy(x_window_scaled.astype(np.float32))
        
        # Get target data for both time t and t+1
        target_t_np = self.targets_t[idx]
        target_t_plus_1_np = self.targets_t_plus_1[idx]
        
        physics_data_t = {key: torch.from_numpy(val.astype(np.float32)) for key, val in target_t_np.items()}
        # Only need dof_vel for t+1, but we can package it nicely
        physics_data_t['dof_vel_t_plus_1'] = torch.from_numpy(target_t_plus_1_np['dof_vel'].astype(np.float32))
        
        physics_data_t['nn_input'] = nn_input
        return physics_data_t

# --- 4. Custom Collate Function (Unchanged) ---
def collate_fn_physics(batch):
    collated_batch = {}
    if not batch: return collated_batch
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated_batch

# --- 5. Training Loop (MODIFIED for Hybrid Loss) ---
def train(model, train_loader, val_loader, criterion,mse_criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience, dt, loss_weight_vel):
    best_model_path = os.path.join(session_save_dir, 'best_params_predictor.pth')
    min_val_loss = float('inf')
    epochs_no_improve = 0
    r_tensor = torch.tensor([
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 
        0.025, 
        0.025, 0.025, 0.025, 0.025, 0.018, 
        0.025, 0.025, 0.025, 0.025, 0.018
    ], dtype=torch.float32).to(device)

    print(f"\n--- Starting Training with Hybrid Loss (w_vel={loss_weight_vel}, dt={dt}) ---")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            nn_input = batch_on_device['nn_input']
            optimizer.zero_grad()
            F_trans, J, tao_load = model(nn_input)
            
            # --- HYBRID LOSS CALCULATION ---
            # 1. Primary Loss: Acceleration at time t
            accel_pred_t = physics_compute_dof_alpha(
                f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                torque_a=batch_on_device['torque'], tao_load=tao_load, 
                f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
            )
            loss_accel = criterion(accel_pred_t, batch_on_device['dof_angular_acceleration'])

            # 2. Auxiliary Loss: Integrated Velocity at time t+1
            vel_true_t = batch_on_device['dof_vel']
            vel_pred_t_plus_1 = vel_true_t + accel_pred_t * dt # Euler integration
            vel_true_t_plus_1 = batch_on_device['dof_vel_t_plus_1']
            
            loss_vel = criterion(vel_pred_t_plus_1, vel_true_t_plus_1)
            #loss_vel = mse_criterion(vel_pred_t_plus_1, vel_true_t_plus_1)
            
            # 3. Combine losses
            total_loss =  loss_accel + loss_weight_vel * loss_vel
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += total_loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop (should also be updated with the same hybrid loss logic)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # ... (calculate total_loss for validation set similarly) ...
                batch_on_device = {k: v.to(device) for k, v in batch.items()}
                nn_input = batch_on_device['nn_input']
                F_trans, J, tao_load = model(nn_input)
                
                accel_pred_t = physics_compute_dof_alpha(
                    f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                    torque_a=batch_on_device['torque'], tao_load=tao_load, 
                    f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
                )
                loss_accel_val = criterion(accel_pred_t, batch_on_device['dof_angular_acceleration'])

                vel_true_t = batch_on_device['dof_vel']
                vel_pred_t_plus_1 = vel_true_t + accel_pred_t * dt
                vel_true_t_plus_1 = batch_on_device['dof_vel_t_plus_1']

                loss_vel_val = criterion(vel_pred_t_plus_1, vel_true_t_plus_1)
                #loss_vel_val = mse_criterion(vel_pred_t_plus_1, vel_true_t_plus_1)
                
                total_loss_val = loss_accel_val + loss_weight_vel * loss_vel_val
                total_val_loss += total_loss_val.item()

        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")
        print(f'loss_accel_val:{loss_accel_val}, loss_vel_val(scaled):{loss_vel_val * loss_weight_vel}')
        # ... (saving model, early stopping, scheduler step) ...
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

# --- 6. Main Execution Block (MODIFIED) ---
def main():
    # --- Configuration ---
    input_keys = ['base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'dof_angular_acceleration']
    num_dofs = 23
    window_size = 10 
    input_dim_per_frame = sum([3, 3, 23, 23, 23]) # 75
    
    # ***** Architecture Configuration *****
    cnn_out_channels = 256
    block_dims = [512, 512] 
    
    # ***** Training Hyperparameters *****
    # NEW: dt for integration and weight for hybrid loss
    dt = 1 / 200 # isaacgym sim fps 200
    loss_weight_vel = 100 # Weight for the velocity loss term. 

    learning_rate = 1e-4 #1e-4#1e-3 
    weight_decay = 5e-2
    batch_size = 4096 
    num_epochs = 50000
    patience = 100
    scheduler_patience = 20
    dropout_rate = 0.0

    # --- Paths and Session Management ---
    log_base_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs'
    data_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/data_for_training'
    
    experiment_name = 'dev_resnet_TemporalCNN' 
    for arg in sys.argv:
        if arg.startswith('+experiment='):
            experiment_name = arg.split('=')[1]
            break
            
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_save_dir = os.path.join(log_base_dir, f"{timestamp}_{experiment_name}")
    os.makedirs(session_save_dir, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Standardization (MODIFIED for windowed data) ---
    print("Loading pre-split data...")
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True).item()
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True).item()

    print("Fitting StandardScaler on training data...")
    sorted_input_keys = sorted(input_keys)
    
    # Correctly fit the scaler on the per-frame feature distribution
    full_train_x_raw = np.concatenate([train_data[key] for key in sorted_input_keys], axis=2)
    num_features = full_train_x_raw.shape[2]
    full_train_x_reshaped = full_train_x_raw.reshape(-1, num_features)
    
    x_scaler = StandardScaler()
    x_scaler.fit(full_train_x_reshaped)
    del full_train_x_raw, full_train_x_reshaped 
    
    joblib.dump(x_scaler, os.path.join(session_save_dir, 'x_scaler.gz'))
    print(f"Input feature scaler (x_scaler) saved to {session_save_dir}")

    # --- Create Datasets and DataLoaders (MODIFIED) ---
    train_dataset = PhysicsDataset(train_data, input_keys, x_scaler, window_size)
    val_dataset = PhysicsDataset(val_data, input_keys, x_scaler, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_physics, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_physics, num_workers=4, pin_memory=True)

    # --- Initialize Model, Criterion, and Optimizer ---
    model = TemporalParameterNet(input_dim_per_frame=input_dim_per_frame, 
                                 cnn_out_channels=cnn_out_channels,
                                 block_dims=block_dims, 
                                 num_dofs=num_dofs, 
                                 dropout_rate=dropout_rate).to(device)
    
    criterion = nn.SmoothL1Loss()
    mse_criterion = nn.MSELoss()
    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-9)


    print("\n--- Starting run for experiment:", experiment_name, "---")
    print(f"Input dim per frame: {input_dim_per_frame} | Window: {window_size} -> Model Type: Temporal CNN")
    print(f"Network Architecture (CNN out channels): {cnn_out_channels}")
    print(f"Network Architecture (MLP block_dims): {block_dims}")
    print("\nModel Structure:")
    print(model)
    
    # --- Start Training ---
    # Pass new hyperparams to the train function
    train(model, train_loader, val_loader, criterion, mse_criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience, dt, loss_weight_vel)

if __name__ == '__main__':
    main()