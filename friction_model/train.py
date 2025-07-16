import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import datetime
import sys

# --- 1. Differentiable Physics Model (PyTorch Version) - No changes needed ---
def differentiable_physics_model(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    """
    Differentiable physics model implemented in PyTorch.
    All inputs are expected to be PyTorch Tensors.
    """
    sign_dof_vel = torch.sign(dof_vel)
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    dof_vel_pre = numerator / (f_v + 1e-8)
    return dof_vel_pre

# --- 2. The "Model" with Parameter Scaling ---
class SystemIdentifier(nn.Module):
    def __init__(self, num_dofs=23, param_scales={'F_trans': 1.0, 'J': 0.01, 'tao_load': 1.0}):
        super(SystemIdentifier, self).__init__()
        
        # We learn the "raw" (normalized) parameters.
        # Initialize them near zero to let the optimizer find their scale.
        self.F_trans_raw = nn.Parameter(torch.randn(num_dofs) * 0.1)
        self.J_raw = nn.Parameter(torch.randn(num_dofs) * 0.1)
        self.tao_load_raw = nn.Parameter(torch.randn(num_dofs) * 0.1)
        
        # Store scale factors as non-trainable buffers
        self.register_buffer('f_trans_scale', torch.tensor(param_scales['F_trans'], dtype=torch.float32))
        self.register_buffer('j_scale', torch.tensor(param_scales['J'], dtype=torch.float32))
        self.register_buffer('tao_load_scale', torch.tensor(param_scales['tao_load'], dtype=torch.float32))

        # Register 'r' as a fixed constant buffer
        r_values = torch.tensor([
            0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 0.025, 0.025, 0.025, 
            0.025, 0.008, 0.006, 0.025, 0.025, 0.025, 0.025, 0.025, 0.018, 
            0.025, 0.025, 0.025, 0.025, 0.018
        ], dtype=torch.float32)
        self.register_buffer('r', r_values)

        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        # Scale the raw learned parameters to their physical values before using them
        F_trans_physical = self.F_trans_raw * self.f_trans_scale
        J_physical = self.J_raw * self.j_scale
        tao_load_physical = self.tao_load_raw * self.tao_load_scale
        
        # Unpack batch data
        dof_vel, f_c, dof_ang_acc, torque_a, f_v = (
            batch['dof_vel'], batch['friction_coeffs'], batch['dof_angular_acceleration'],
            batch['torque'], batch['viscous_friction_coeffs']
        )
        
        # Predict dof_vel using the scaled physical parameters
        dof_vel_predicted = differentiable_physics_model(
            f_c, self.r, F_trans_physical, J_physical, dof_ang_acc, 
            torque_a, tao_load_physical, f_v, dof_vel
        )
        
        loss = self.loss_fn(dof_vel_predicted, dof_vel)
        return loss

# --- 3. Custom Dataset and Collate Function (No changes needed) ---
class PhysicsDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.data_keys = sorted(list(data.keys()))
        self.num_iterations = len(data[self.data_keys[0]])

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx):
        return {key: torch.from_numpy(self.data[key][idx].astype(np.float32)) for key in self.data_keys}

def collate_fn_physics(batch):
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [item[key] for item in batch]
        collated_batch[key] = torch.cat(tensors, dim=0)
    return collated_batch

# --- 4. Training Loop (No changes needed) ---
def train(model, train_loader, val_loader, optimizer, scheduler, device, session_save_dir, num_epochs, patience):
    best_model_path = os.path.join(session_save_dir, 'best_system_params.pth')
    min_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"\n--- Starting System Identification ---")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(batch_on_device)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = {k: v.to(device) for k, v in batch.items()}
                loss = model(batch_on_device)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best validation loss. Parameters saved.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break
        scheduler.step(avg_val_loss)

    print("\n--- Identification Finished ---")
    print(f"Best parameters saved at: {best_model_path}")
    
    # Load and print the final, scaled physical parameters
    best_model_state = torch.load(best_model_path)
    final_F_trans = best_model_state['F_trans_raw'].cpu().numpy() * model.f_trans_scale.cpu().numpy()
    final_J = best_model_state['J_raw'].cpu().numpy() * model.j_scale.cpu().numpy()
    final_tao_load = best_model_state['tao_load_raw'].cpu().numpy() * model.tao_load_scale.cpu().numpy()
    
    print("\nIdentified Physical Parameters:")
    print("F_trans:", final_F_trans)
    print("J:", final_J)
    print("tao_load:", final_tao_load)

# --- 5. Main Execution Block ---
def main():
    # --- Config ---
    learning_rate = 0.01
    weight_decay = 1e-5
    batch_size = 64
    num_epochs = 1000
    patience = 50
    scheduler_patience = 10
    param_scales = {'F_trans': 1.0, 'J': 0.01, 'tao_load': 1.0}

    # --- Paths and Session ---
    log_base_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs'
    data_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/LOOSE_JumpJumpJump_RandViscous_Hard_concat_data/data_for_training'
    
    experiment_name = 'dev_friction_model' # Default experiment name
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

    # --- Data ---
    print("Loading pre-split data...")
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True).item()
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True).item()
    
    train_dataset = PhysicsDataset(train_data)
    val_dataset = PhysicsDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_physics)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_physics)

    # --- Model and Optimizer ---
    model = SystemIdentifier(num_dofs=23, param_scales=param_scales).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience)
    
    print("\n--- Starting run for experiment:", experiment_name, "---")
    print("Parameter scales being used:", param_scales)
    
    # --- Start Training ---
    train(model, train_loader, val_loader, optimizer, scheduler, device, session_save_dir, num_epochs, patience)

if __name__ == '__main__':
    main()