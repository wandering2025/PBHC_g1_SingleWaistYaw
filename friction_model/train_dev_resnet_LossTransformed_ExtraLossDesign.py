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

# --- 1a. Residual Block Definition ---
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

# --- 1b. Parameter Prediction Network with ResNet Architecture ---
class ParameterNetResNet(nn.Module):
    """
    A ResNet-based MLP that predicts physical parameters from state vectors.
    This architecture uses a stack of residual blocks for robust feature extraction.
    """
    def __init__(self, input_dim: int, block_dims: list, num_dofs: int = 23, dropout_rate: float = 0.1):
        """
        Initializes the ResNet-based network.
        :param input_dim: Dimension of the input feature vector (e.g., 98).
        :param block_dims: A list defining the output dimension of each of the 3 residual blocks.
                           Example: [1024, 512, 256] creates 3 blocks with these output dimensions.
        :param num_dofs: Number of degrees of freedom (e.g., 23).
        :param dropout_rate: Dropout probability.
        """
        super(ParameterNetResNet, self).__init__()
        #assert len(block_dims) == 3, "This network is configured for exactly 3 residual blocks."
        
        self.num_dofs = num_dofs
        
        # Initial layer to project input to the first block's dimension
        first_block_dim = block_dims[0]
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, first_block_dim),
            nn.BatchNorm1d(first_block_dim),
            nn.ReLU()
        )
        
        # # Stack of 3 Residual Blocks
        # self.res_block1 = ResidualBlock(first_block_dim, block_dims[0], dropout_rate)
        # self.res_block2 = ResidualBlock(block_dims[0], block_dims[1], dropout_rate)
        # self.res_block3 = ResidualBlock(block_dims[1], block_dims[2], dropout_rate)

        res_layers = []
        current_dim = first_block_dim
        for h_dim in block_dims:
            res_layers.append(ResidualBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim # Update the dimension for the next block
            
        self.residual_blocks = nn.Sequential(*res_layers)

        
        # Final output head (same as before)
        #self.output_head = nn.Linear(block_dims[2], num_dofs * 3)
        self.output_head = nn.Linear(current_dim, num_dofs * 3)

    def forward(self, x: torch.Tensor):
        # 1. Pass through the initial projection layer
        x = self.initial_layer(x)
        
        # 2. Pass through the stack of 3 residual blocks
        # x = self.res_block1(x)
        # x = self.res_block2(x)
        # x = self.res_block3(x)

        x = self.residual_blocks(x)
        
        # 3. Predict the raw parameter values using the final head
        raw_params = self.output_head(x)
        
        # 4. Split and process parameters (this part remains unchanged)
        f_trans_raw, j_raw, tao_load_raw = torch.split(raw_params, self.num_dofs, dim=1)
        
        F_trans = f_trans_raw
        tao_load = tao_load_raw
        J = F.softplus(torch.tanh(j_raw))
        
        return F_trans, J, tao_load

# --- 2. Differentiable Physics Model (Unchanged) ---
def differentiable_physics_model(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    """
    Differentiable physics model implemented in PyTorch.
    All inputs are expected to be PyTorch Tensors.
    """
    sign_dof_vel = torch.sign(dof_vel)
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    #dof_vel_pre = numerator / (f_v + 1e-8)
    #return dof_vel_pre
    return numerator

# --- 3. Custom Dataset with Standardization (Unchanged) ---
class PhysicsDataset(Dataset):
    def __init__(self, data: dict, input_keys: list, x_scaler: StandardScaler):
        self.data = data
        self.input_keys = sorted(input_keys)
        self.all_keys = sorted(list(data.keys()))
        self.num_iterations = len(data[self.all_keys[0]])
        self.x_scaler = x_scaler

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx: int):
        x_iter_raw = np.concatenate([self.data[key][idx] for key in self.input_keys], axis=1, dtype=np.float32)
        x_iter_scaled = self.x_scaler.transform(x_iter_raw)
        physics_data = {key: torch.from_numpy(self.data[key][idx].astype(np.float32)) for key in self.all_keys}
        physics_data['nn_input'] = torch.from_numpy(x_iter_scaled)
        return physics_data

# --- 4. Custom Collate Function (Unchanged) ---
def collate_fn_physics(batch):
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [item[key] for item in batch]
        collated_batch[key] = torch.cat(tensors, dim=0)
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

    numerator_threshhold = 4.0
    dof_vel_threshhold = 40.0

    boundary_loss_weight = 1.0 


    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        
        
        for batch in train_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            nn_input = batch_on_device['nn_input']
            optimizer.zero_grad()
            F_trans, J, tao_load = model(nn_input)
            
            numerator_pre = differentiable_physics_model(
            #dof_vel_predicted = differentiable_physics_model(
                f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
            )
            
            target_numerator = batch_on_device['viscous_friction_coeffs'] * batch_on_device['dof_vel']

            #norm_factor = torch.abs(batch_on_device['torque']) + 1.0
            norm_factor = 1.0

            loss_cri = criterion(numerator_pre / norm_factor, target_numerator / norm_factor)

            #loss_mse = criterion(numerator_pre, target_numerator)
            #loss = criterion(dof_vel_predicted, batch_on_device['dof_vel'])
            dof_vel_pre = numerator_pre / (batch_on_device['viscous_friction_coeffs'] + 1e-8)


            # Sure that dof_vel is within limit [-37,37] and viscous_friction_coeffs is normally within [0,0.1]
            boundary_violations = F.relu(torch.abs(numerator_pre) - numerator_threshhold) \
                + F.relu(torch.abs(dof_vel_pre) - dof_vel_threshhold)
            loss_boundary = torch.mean(boundary_violations)

            total_loss = loss_cri + boundary_loss_weight * loss_boundary

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += total_loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = {k: v.to(device) for k, v in batch.items()}
                nn_input = batch_on_device['nn_input']
                F_trans, J, tao_load = model(nn_input)
                numerator = differentiable_physics_model(
                #dof_vel_predicted = differentiable_physics_model(
                    f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                    dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                    tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
                )

                pre_dof_vel = numerator / (batch_on_device['viscous_friction_coeffs'] + 1e-8)

                target_numerator = batch_on_device['viscous_friction_coeffs'] * batch_on_device['dof_vel']

                #norm_factor_val = torch.abs(batch_on_device['torque']) + 1.0
                norm_factor_val = 1.0

                loss_cri_val = criterion(numerator / norm_factor_val, target_numerator / norm_factor_val)
                #loss_mse_val = criterion(numerator, target_numerator)

                boundary_violations_val = F.relu(torch.abs(numerator) - numerator_threshhold) \
                                        + F.relu(torch.abs(pre_dof_vel) - dof_vel_threshhold)
                loss_boundary_val = torch.mean(boundary_violations_val)
                
                total_loss_val = loss_cri_val + boundary_loss_weight * loss_boundary_val
                
                total_val_loss += total_loss_val.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")

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

# --- 6. Main Execution Block (MODIFIED TO USE NEW NETWORK) ---
def main():
    # --- Configuration ---
    input_keys = ['base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'dof_angular_acceleration', 'torque']
    input_dim = sum([3, 3, 23, 23, 23, 23]) # 98
    num_dofs = 23
    
    # ***** ResNet Architecture Configuration *****

    block_dims = [512, 1024,1024, 512] 
    
    # Training Hyperparameters
    learning_rate = 1e-3 #1e-4
    weight_decay = 0.0
    batch_size = 1024
    num_epochs = 50000
    patience = 150
    scheduler_patience = 40
    dropout_rate = 0.0 # Dropout for residual blocks

    # --- Paths and Session Management ---
    log_base_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs'
    data_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/LOOSE_JumpJumpJump_RandViscous_Hard_concat_data/data_for_training'
    
    experiment_name = 'resnet_LossTransformed_ExtraLossDesign_L1Loss' # Default experiment name
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

    # --- Data Loading and Standardization ---
    print("Loading pre-split data...")
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True).item()
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True).item()

    print("Fitting StandardScaler on training data...")
    sorted_input_keys = sorted(input_keys)
    full_train_x_list = [np.concatenate([train_data[key][i] for key in sorted_input_keys], axis=1) for i in range(len(train_data[input_keys[0]]))]
    full_train_x = np.concatenate(full_train_x_list, axis=0)
    
    x_scaler = StandardScaler()
    x_scaler.fit(full_train_x)
    del full_train_x, full_train_x_list
    
    joblib.dump(x_scaler, os.path.join(session_save_dir, 'x_scaler.gz'))
    print(f"Input feature scaler (x_scaler) saved to {session_save_dir}")

    # --- Create Datasets and DataLoaders ---
    train_dataset = PhysicsDataset(train_data, input_keys, x_scaler)
    val_dataset = PhysicsDataset(val_data, input_keys, x_scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_physics)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_physics)

    # --- Initialize Model, Criterion, and Optimizer ---
    # USE THE NEW RESNET MODEL
    model = ParameterNetResNet(input_dim=input_dim, 
                               block_dims=block_dims, 
                               num_dofs=num_dofs, 
                               dropout_rate=dropout_rate).to(device)
    
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    
    print("\n--- Starting run for experiment:", experiment_name, "---")
    print(f"Network Architecture (block_dims): {block_dims}")
    print("\nModel Structure:")
    print(model)
    
    # --- Start Training ---
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience)

if __name__ == '__main__':
    main()