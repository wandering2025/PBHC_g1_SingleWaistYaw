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
    """
    A residual block with a skip connection.
    If input and output dimensions are different, a linear layer
    is used on the shortcut path to match the dimensions.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        # Main path of the block
        self.main_path = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        # Shortcut connection to handle dimension changes
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output is the sum of the main path and the shortcut path
        return F.relu(self.main_path(x) + self.shortcut(x))

# --- 1b. Parameter Prediction Network with ResNet Architecture (Unchanged) ---
class ParameterNetResNet(nn.Module):
    """
    A ResNet-based MLP that predicts physical parameters from state vectors.
    This architecture uses a stack of residual blocks for robust feature extraction.
    """
    def __init__(self, input_dim: int, block_dims: list, num_dofs: int = 23, dropout_rate: float = 0.1):
        super(ParameterNetResNet, self).__init__()
        
        self.num_dofs = num_dofs
        
        # Initial layer to project input to the first block's dimension
        first_block_dim = block_dims[0]
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, first_block_dim),
            nn.BatchNorm1d(first_block_dim),
            nn.ReLU()
        )
        
        # Stack of Residual Blocks
        res_layers = []
        current_dim = first_block_dim
        for h_dim in block_dims:
            res_layers.append(ResidualBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim # Update the dimension for the next block
            
        self.residual_blocks = nn.Sequential(*res_layers)

        # Final output head
        self.output_head = nn.Linear(current_dim, num_dofs * 3)

    def forward(self, x: torch.Tensor):
        # 1. Pass through the initial projection layer
        x = self.initial_layer(x)
        
        # 2. Pass through the stack of residual blocks
        x = self.residual_blocks(x)
        
        # 3. Predict the raw parameter values using the final head
        raw_params = self.output_head(x)
        
        # 4. Split and process parameters
        f_trans_raw, j_raw, tao_load_raw = torch.split(raw_params, self.num_dofs, dim=1)
        
        F_trans = f_trans_raw
        tao_load = tao_load_raw
        J = F.softplus(torch.tanh(j_raw))
        
        return F_trans, J, tao_load

# --- 2. Differentiable Physics Model (Unchanged) ---
def physics_compute_dof_alpha(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    """
    Differentiable physics model implemented in PyTorch.
    All inputs are expected to be PyTorch Tensors.
    """
    sign_dof_vel = torch.sign(dof_vel)
    dof_ang_acc__mult__J = -f_c * r * F_trans * sign_dof_vel - f_v * dof_vel + torque_a + tao_load
    dof_ang_acc_pre = dof_ang_acc__mult__J / (J + 1e-8)
    return dof_ang_acc_pre


def physics_compute_dof_vel(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    """
    Differentiable physics model implemented in PyTorch.
    All inputs are expected to be PyTorch Tensors.
    """
    sign_dof_vel = torch.sign(dof_vel)
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    dof_vel_pre = numerator / (f_v + 1e-8)
    return dof_vel_pre

# --- 3. Custom Dataset with Time Window (MODIFIED) ---
class PhysicsDataset(Dataset):
    """
    Custom dataset that creates sliding windows of historical data.
    Each sample consists of a window of input features and the target values
    at the final timestep of that window.
    """
    def __init__(self, data: dict, input_keys: list, x_scaler: StandardScaler, window_size: int):
        self.input_keys = sorted(input_keys)
        self.all_keys = sorted(list(data.keys()))
        self.x_scaler = x_scaler
        self.window_size = window_size
        
        print(f'------------INIT Dataset with Window Size: {self.window_size}---------------------')
        
        # This helper function will create all the valid sliding windows
        self._create_windows(data)

    def _create_windows(self, data: dict):
        """
        Generates input windows and corresponding targets from the raw data.
        This ensures that windows do not cross iteration boundaries.
        """
        # Concatenate all specified input features into one large array
        # Shape: (num_iterations, sequence_length, feature_dim_per_frame)
        x_raw_all = np.concatenate([data[key] for key in self.input_keys], axis=2, dtype=np.float32)
        
        num_iterations, seq_len, _ = x_raw_all.shape
        
        self.windows = []
        self.targets = []
        
        print(f"Generating sliding windows... Num iterations: {num_iterations}, Seq length: {seq_len}")
        
        # Iterate through each iteration to create sliding windows
        for i in range(num_iterations):
            # A valid window requires `window_size` frames.
            # So, the last frame of the first window is at index `window_size - 1`.
            for end_frame in range(self.window_size - 1, seq_len):
                start_frame = end_frame - self.window_size + 1
                
                # Extract the input window from the concatenated features
                # Shape: (window_size, feature_dim_per_frame)
                input_window = x_raw_all[i, start_frame:end_frame + 1, :]
                self.windows.append(input_window)
                
                # The target data corresponds to the state at the `end_frame`
                target_data_at_t = {key: data[key][i, end_frame] for key in self.all_keys}
                self.targets.append(target_data_at_t)
                
        print(f"Successfully created {len(self.windows)} total samples from the data.")

    def __len__(self):
        # The length of the dataset is the total number of windows created
        return len(self.windows)

    def __getitem__(self, idx: int):
        # Get the raw input window for the given index
        x_window_raw = self.windows[idx]
        
        # Scale the features for the window. The scaler was fitted on (n_samples, n_features).
        x_window_scaled = self.x_scaler.transform(x_window_raw)
        
        # Flatten the window (e.g., 10x75) to create a single feature vector (750,) for the MLP
        nn_input_flat = torch.from_numpy(x_window_scaled.flatten().astype(np.float32))
        
        # Get the corresponding target data (from the last frame) and convert to tensors
        target_data_np = self.targets[idx]
        physics_data = {key: torch.from_numpy(val.astype(np.float32)) for key, val in target_data_np.items()}
        
        # Add the flattened, scaled NN input to the dictionary to be returned
        physics_data['nn_input'] = nn_input_flat
        
        return physics_data

# --- 4. Custom Collate Function (Unchanged) ---
def collate_fn_physics(batch):
    """
    This collate function takes a list of dictionary-based samples
    and stacks them into a single batch dictionary.
    This works perfectly with our new dataset structure.
    """
    collated_batch = {}
    if not batch:
        return collated_batch
    keys = batch[0].keys()
    for key in keys:
        tensors = [item[key] for item in batch]
        # Stacks list of tensors into a single tensor for the batch
        collated_batch[key] = torch.stack(tensors, dim=0)
    return collated_batch

# --- 5. Training Loop (Unchanged) ---
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience):
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

    print(f"\n--- Starting Training: State-Dependent Parameter Prediction ---")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            nn_input = batch_on_device['nn_input']
            optimizer.zero_grad()
            F_trans, J, tao_load = model(nn_input)

            dof_ang_acc_pre = physics_compute_dof_alpha(
                f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
            )

            dof_vel_pre = physics_compute_dof_vel(
                f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
            )

            loss_dof_alpha = criterion(dof_ang_acc_pre, batch_on_device['dof_angular_acceleration'])
            loss_dof_vel =  criterion(dof_vel_pre, batch_on_device['dof_vel'])

            loss = loss_dof_alpha + loss_dof_vel

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = {k: v.to(device) for k, v in batch.items()}
                nn_input = batch_on_device['nn_input']
                F_trans, J, tao_load = model(nn_input)

                dof_ang_acc_pre_val = physics_compute_dof_alpha(
                    f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                    dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                    tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
                )

                dof_vel_pre_val = physics_compute_dof_vel(
                    f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                    dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                    tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
                )

                loss_dof_alpha_val = criterion(dof_ang_acc_pre_val, batch_on_device['dof_angular_acceleration'])
                loss_dof_vel_val = criterion(dof_vel_pre_val, batch_on_device['dof_vel'])
                loss_val = loss_dof_alpha_val + loss_dof_vel_val

                total_val_loss += loss_val.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")
        print(f'---------------------- loss_dof_alpha_val: {loss_dof_alpha_val:.6f} | loss_dof_vel_val: {loss_dof_vel_val:.6f}-------------------')

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
    
    # NEW: Configuration for time window
    window_size = 10 
    
    # UPDATE: Calculate input dimension based on window size
    input_dim_per_frame = sum([3, 3, 23, 23, 23]) # 75 features per frame
    input_dim = input_dim_per_frame * window_size  # Total input dim is features * window_size
    
    # ***** ResNet Architecture Configuration *****
    block_dims = [512,512,512]#[512,216]
    
    # Training Hyperparameters
    learning_rate = 1e-3#5e-4 #1e-3
    weight_decay = 1e-3
    batch_size = 4096 
    num_epochs = 50000
    patience = 150
    scheduler_patience = 40
    dropout_rate = 0.2

    # --- Paths and Session Management ---
    log_base_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs'
    # UPDATE: New data directory
    data_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/data_for_training'
    
    experiment_name = 'dev_resnet_PredDofAngAcce_Window10' # Changed experiment name to reflect window size
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

    # --- Data Loading and Standardization (MODIFIED) ---
    print("Loading pre-split data...")
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True).item()
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True).item()

    print("Fitting StandardScaler on training data...")
    sorted_input_keys = sorted(input_keys)
    print(f"Sorted Input keys for scaler: {sorted_input_keys}")

    # Concatenate all input features for the training set along the feature dimension
    # Shape: (num_iterations, seq_len, feature_dim_per_frame)
    full_train_x_raw = np.concatenate([train_data[key] for key in sorted_input_keys], axis=2)

    # Reshape to (total_timesteps, feature_dim_per_frame) to fit the scaler.
    # This correctly fits the scaler on the distribution of each feature across all time steps.
    num_features = full_train_x_raw.shape[2]
    full_train_x_reshaped = full_train_x_raw.reshape(-1, num_features)
    
    x_scaler = StandardScaler()
    x_scaler.fit(full_train_x_reshaped)
    del full_train_x_raw, full_train_x_reshaped # Free up memory
    
    joblib.dump(x_scaler, os.path.join(session_save_dir, 'x_scaler.gz'))
    print(f"Input feature scaler (x_scaler) saved to {session_save_dir}")

    # --- Create Datasets and DataLoaders (MODIFIED) ---
    # Pass the window_size to the dataset constructor
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    
    print("\n--- Starting run for experiment:", experiment_name, "---")
    print(f"Input dim per frame: {input_dim_per_frame} | Window: {window_size} -> Total Input Dim: {input_dim}")
    print(f"Network Architecture (block_dims): {block_dims}")
    print("\nModel Structure:")
    print(model)
    
    # --- Start Training ---
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience)

if __name__ == '__main__':
    main()